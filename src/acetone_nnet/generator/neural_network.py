"""Code generation module of ACETONE.

*******************************************************************************
* ACETONE: Predictable programming framework for ML applications in safety-critical systems
* Copyright (c) 2022. ONERA
* This file is part of ACETONE
*
* ACETONE is free software ;
* you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation ;
* either version 3 of  the License, or (at your option) any later version.
*
* ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License along with this program ;
* if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
******************************************************************************.
"""

import json
import logging
from abc import ABC
from pathlib import Path
from sys import version_info
from typing import Any

import numpy as np
import onnx
import pystache

if (3, 12) >= version_info >= (3, 10):
    from keras.engine.functional import Functional
    from keras.engine.sequential import Sequential
from typing_extensions import Self

from acetone_nnet import templates
from acetone_nnet.importers.parser import parser
from acetone_nnet.quantize import qform
from acetone_nnet.quantize.activation import QuantizeShiftActivation
from acetone_nnet.templates.template_makefile import TemplateMakefile
from acetone_nnet.versioning.versioning import versioning

from .layers import (
    AveragePooling2D,
    BatchNormalization,
    Broadcast,
    Concatenate,
    ConstantLayer,
    Conv2D,
    Conv2D6loops,
    Conv2DGemmTarget,
    Conv2DIndirectGemm,
    Conv2DStdGemm,
    Dense,
    Gather,
    GatherElements,
    Gemm,
    MatMul,
    MaxPooling2D,
    Pooling2D,
    Reduce,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Softmax,
)

MODEL_TYPE = str | Path | onnx.ModelProto
if (3, 12) > version_info >= (3, 10):
    MODEL_TYPE = MODEL_TYPE | Sequential | Functional


class CodeGenerator(ABC):
    """Main module of ACETONE."""

    def __init__(
        self: Self,
        file: MODEL_TYPE,
        test_dataset: str | np.ndarray | Path | None = None,
        external_input: bool | None = False,
        function_name: str = "inference",
        target: str = "generic",
        target_page_size: int = 4096,
        nb_tests: int | str = 0,
        versions: dict[int | str, str] | None = None,
        normalize: bool | str = False,
        debug_mode: str | None = None,
        verbose: bool = False,
        to_hex: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the class."""
        self.file = file
        self.function_name = function_name
        self.normalize = bool(normalize)
        self.template_path = templates.__file__[:-11]
        self.verbose = verbose
        self.to_hex = to_hex
        self.target_cfg = None
        if target != "generic":
            try:
                self.target_cfg = json.loads(
                    (Path((target + ".json"))).read_text(),
                )
                self.to_hex = False
                logging.info(f'Target configuration {self.target_cfg["name"]} loaded')
            except json.JSONDecodeError as e:
                logging.warning(
                    f'Target configuration file {target+".json"} parse error: {e}',
                )
            except FileNotFoundError:
                logging.warning(
                    f'Target configuration file {target+".json"} not found, continue',
                )

        if not self.normalize:
            l, dtype, dtype_py, data_format, maxpath, dict_cst = parser(
                file_to_parse=self.file,
            )
        else:  # normalise
            l, dtype, dtype_py, data_format, maxpath, dict_cst, normalizer = parser(
                file_to_parse=self.file,
                normalize=self.normalize,
            )
            self.Normalizer = normalizer

        self.default_implementations = {
            "Conv2D": "6loops",
        }
        self.layers: list[Any] = l
        self.versions = self.select_layers_implementation(versions)
        self.layers = versioning(self.layers, self.versions)
        self.data_type = (
            self.target_cfg["quantization"]["dtype"]
            if self.target_cfg is not None
            else dtype
        )
        self.data_type_py = (
            np.dtype(self.target_cfg["quantization"]["pydtype"])
            if self.target_cfg is not None
            else dtype_py
        )
        logging.info(f"C type {self.data_type}")
        logging.info(f"py dtype {self.data_type_py}")
        self.quantize_layers()
        self.maxpath = maxpath
        self.data_format = data_format
        self.dict_cst = dict_cst

        self.read_ext_input = external_input
        self.nb_tests = int(nb_tests)

        self.test_dataset = self._initialise_dataset(test_dataset, int(nb_tests))

        self.files_to_gen = [
            "inference.c",
            "inference.h",
            "global_vars.c",
            "main.c",
            "Makefile",
            "test_dataset.h",
        ]
        if not self.read_ext_input:
            self.files_to_gen.append("test_dataset.c")

        self.target = target
        if self.target != "generic":
            self.files_to_gen.append("target.c")
            self.files_to_gen.append("target.h")

        self.target_page_size = target_page_size

        ##### Debug Mode #####
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_target = self.load_debug_target(debug_mode)
        ##### Debug Mode #####

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if not isinstance(self.file, MODEL_TYPE):
            msg = "Error: model type.\n Format must be: path to model, model ONNX or model Keras"
            raise TypeError(msg)
        if not (
            isinstance(test_dataset, str | np.ndarray | Path) or test_dataset is None
        ):
            msg = "Error: test_dataset type.\n Must be: path to dataset text file, numpy array or None"
            raise TypeError(msg)
        if not isinstance(self.function_name, str):
            msg = "Error: function name type.\n Must be a string"
            raise TypeError(msg)
        if not (self.debug_mode is None or isinstance(self.debug_mode, str)):
            msg = "Error: debug mode type.\n Must be: string or None"
            raise TypeError(msg)
        if not isinstance(self.verbose, bool):
            msg = "Error: verbose type.\n Must be: bool"
            raise TypeError(msg)
        if not (isinstance(self.read_ext_input, bool) or self.read_ext_input is None):
            msg = "Error: external_input typr.\n Must be: bool"
            raise TypeError(msg)

        ### Checking value consistency ###

        if self.read_ext_input and isinstance(test_dataset, str | np.ndarray | Path):
            logging.warning("Warning: given dataset will be ignored")

        # Debug Mode
        if self.debug_mode and self.debug_mode not in ["keras", "onnx", "time"]:
            msg = "Error: debug mode value.\n Must be one of: keras, onnx, time"
            raise ValueError(msg)

    def select_layers_implementation(
        self: Self,
        versions: dict[int | str, str] | None,
    ) -> dict[int, str | None]:
        """Create the dictionary used for the versioning.

        Parameters
        ----------
        versions: dict or None
            Name of selected implementation per layer, or per layer type. If
            None, the default implementation will be used for each layer type.

        Returns
        -------
        dict
            Name of selected implementation for each layer, if not the default one.

        """
        selected_implementations: dict[int, str | None] = {}
        for layer in self.layers:
            # Select the implementation based in priority on layer id, or type
            if versions is not None:
                for k in [layer.idx, layer.name]:
                    if k in versions:
                        selected_implementations[layer.idx] = versions[k]
                        break
                else:
                    # Select the default implementation per layer type, if specified
                    d = self.default_implementations.get(layer.name, None)
                    selected_implementations[layer.idx] = d
        return selected_implementations

    def load_debug_target(
        self: Self,
        debug_mode: str | None,
    ) -> list[int]:
        """Identify list of layers indices to debug."""
        targets: list[int] = []
        for layer in self.layers[1:]:
            if debug_mode == "keras" and layer.name == "Softmax":
                targets[-1] = layer.idx
            else:
                targets.append(layer.idx)

        return targets

    def _initialise_dataset(
        self: Self,
        dataset_or_path: np.ndarray | str | Path | None,
        nb_tests: int,
    ) -> np.ndarray:
        """Initialise dataset for model randomly or from existing data."""
        match dataset_or_path:
            case None:
                # Create random dataset for graph.
                if np.issubdtype(self.data_type_py, np.integer):
                    d = np.random.default_rng(seed=10).integers(
                        np.iinfo(self.data_type_py).min,
                        np.iinfo(self.data_type_py).max,
                        size=(nb_tests, 1, int(self.layers[0].size)),
                        dtype=self.data_type_py,
                    )
                else:
                    d = np.random.default_rng(seed=10).random(
                        size=(nb_tests, 1, int(self.layers[0].size)),
                        dtype=self.data_type_py,
                    )
            case np.ndarray() as dataset:
                d = dataset
            case Path() | str() as path:
                d = self._load_dataset(Path(path), self.data_type_py, nb_tests)
            case _:
                raise ValueError
        return d

    def _load_dataset(
        self: Self,
        path: Path,
        dtype: np.dtype,
        nb_tests: int,
    ) -> np.ndarray:
        """Load a dataset from file.

        Each line of the file holds a separate array formatted as a comma-separated list
        of values surrounded by [ and ] delimiters.

        """
        test_dataset: list[list[np.number]] = []
        with path.open() as f:
            # FIXME This returns at least one array of values even if nb_tests is 0.DONE
            for i, line in enumerate(f):
                if i >= nb_tests:
                    break
                contents = json.loads(line)
                contents = [self.c2data(c) for c in contents]
                test_dataset.append(list(map(dtype, contents)))
        return np.array(test_dataset)

    def get_input_shape(self: Self) -> list[int]:
        """Return the input shape of the model."""
        return self.layers[0].input_shape

    def generate_dataset(self: Self) -> np.ndarray:
        """Generate a dataset for the model."""
        input_shape = self.get_input_shape()

        dataset = np.random.default_rng().random(
            size=(self.nb_tests, *input_shape),
            dtype=self.data_type_py,
        )
        self.test_dataset = dataset
        return dataset

    def compute_inference(
        self: Self,
        c_files_directory: str,
        dataset_or_path: np.ndarray | str | Path | None = None,
    ) -> tuple[list, list] | np.ndarray:
        """Perform inference pass on test dataset."""
        if self.read_ext_input:
            dataset = self._initialise_dataset(
                dataset_or_path=dataset_or_path,
                nb_tests=self.nb_tests,
                data_type=self.data_type_py,
            )
        else:
            dataset = self.test_dataset

        with (Path(c_files_directory) / "output_python.txt").open("w") as fi:
            for nn_input in dataset:
                # Debug Mode output
                debug_output: list[np.ndarray] = []
                targets: list[str] = []

                # Prepare graph input
                if (self.data_format == "channels_last") and (
                    len(self.layers[0].input_shape) == 4
                ):
                    shape = (
                        self.layers[0].input_shape[2],
                        self.layers[0].input_shape[3],
                        self.layers[0].input_shape[1],
                    )
                    nn_input = np.transpose(np.reshape(nn_input, shape), (2, 0, 1))
                if self.normalize:
                    nn_input = self.Normalizer.pre_processing(nn_input)

                # Inputs per layer and predecessor
                # - li[i][j] is an input for i computed from preceding layer j
                # - li[i][i] is the input for layers with no predecessors
                layer_inputs: dict[int, dict[int, np.ndarray]] = {
                    i.idx: {i.idx: nn_input} for i in self.layers
                }

                def gather_inputs(
                    target,
                    target_inputs,
                ) -> np.ndarray | list[np.ndarray]:
                    # Prepare inputs for computation
                    inputs: np.ndarray | list[np.ndarray] = []
                    if not target.previous_layer:
                        inputs.append(target_inputs[target.idx][target.idx])
                    else:
                        inputs.extend(
                            [
                                target_inputs[target.idx][p.idx]
                                for p in target.previous_layer
                            ],
                        )
                    if len(inputs) > 1:
                        return inputs
                    return inputs[0]

                # Forward computation layer by layer
                #   (Assumes layers are topologically sorted)
                for layer in self.layers:
                    # Ensure all inputs of layer are ready
                    if any(
                        p.idx not in layer_inputs[layer.idx]
                        for p in layer.previous_layer
                    ):
                        raise NotImplementedError

                    # Prepare inputs for computation
                    forward_inputs = gather_inputs(layer, layer_inputs)

                    # Compute layer output
                    layer_output = layer.forward_path_layer(forward_inputs)

                    # Write Layer output into successors' input
                    for successor in layer.next_layer:
                        layer_inputs[successor.idx][layer.idx] = layer_output

                    # Cleanup layer inputs
                    del layer_inputs[layer.idx]

                    # Debug Mode
                    if self.debug_mode and layer.idx in self.debug_target:
                        if not isinstance(layer, ConstantLayer):
                            # Add the inference result of the layer to debug_output
                            debug_output.append(layer_output)
                            if (self.data_format == "channels_last") and hasattr(
                                layer,
                                "output_channels",
                            ):
                                debug_output[-1] = np.transpose(
                                    debug_output[-1],
                                    (1, 2, 0),
                                )
                            debug_output[-1] = debug_output[-1].flatten()

                            targets.append(str(layer.name) + " " + str(layer.idx))

                nn_output = layer_output

                # Write results in text files to compare prediction.
                if (self.data_format == "channels_last") and hasattr(
                    layer,
                    "output_channels",
                ):
                    nn_output = np.transpose(nn_output, (1, 2, 0))
                nn_output = np.reshape(nn_output, -1)

                if self.normalize:
                    nn_output = self.Normalizer.post_processing(nn_output)
                out_string = " ".join([self.data2c(n) for n in nn_output])
                print(f"{out_string}", end=" ", file=fi, flush=True)
                print(" ", file=fi)

        logging.info("File output_python.txt generated.")

        if self.debug_mode:
            return debug_output, targets
        return nn_output

    def data2c(self, d):
        if np.issubdtype(self.data_type_py, np.integer):
            return str(d)
        if self.to_hex:
            return float.hex(float(d)).replace("0000000p", "p")
        return str(float(d))

    def c2data(self, c):
        return float.fromhex(c) if self.to_hex else float(c)

    def flatten_array_order_c(self: Self, array: np.ndarray) -> str:
        """Generate C flat array initializer in C order."""
        flattened_aray = array.flatten(order="C")
        s = "\n        {"
        s += ", ".join([self.data2c(d) for d in flattened_aray])
        s += "}"

        return s

    # TODO Scheduled for removal
    def flatten_array(self: Self, array: np.ndarray) -> str:
        """Generate C flat array initializer in Fortran order."""
        s = "\n        {"
        shape = array.shape
        if len(shape) < 4:
            for _i in range(4 - len(shape)):
                shape = (1, *shape)
            array = np.reshape(array, shape)
        s_array = []
        for j in range(shape[3]):
            for k in range(shape[0]):
                for f in range(shape[1]):
                    for i in range(shape[2]):
                        s_array.append(self.data2c(array[k, f, i, j]))
        s += ", ".join(s_array)
        s += "}"
        return s

    def generate_test_dataset_files(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C code for graph test dataset."""
        # Generate test header file
        template = (
            Path(self.template_path) / "template_test_dataset_header.c.tpl"
        ).read_text()
        with Path.open(output_dir / "test_dataset.h", "w+") as test_dataset_header:
            test_dataset_header.write(
                pystache.render(
                    template,
                    {
                        "nb_tests": self.nb_tests,
                        "nb_inputs": self.layers[0].size,
                        "nb_outputs": self.layers[-1].size,
                        "data_type": self.data_type,
                        "read_input": self.read_ext_input,
                    },
                ),
            )

        if not self.read_ext_input:
            # Generate test source file
            dataset = "{"
            if self.test_dataset is not None:
                dataset += ",".join(
                    map(self.flatten_array_order_c, self.test_dataset),
                )
            dataset += "};\n"

            template = (
                Path(self.template_path) / "template_test_dataset_source.c.tpl"
            ).read_text()
            with Path.open(output_dir / "test_dataset.c", "w+") as test_dataset_source:
                test_dataset_source.write(
                    pystache.render(
                        template,
                        {"data_type": self.data_type, "dataset": dataset},
                    ),
                )

    def generate_main_file(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate entry point C code."""
        template = (Path(self.template_path) / "template_main_file.c.tpl").read_text()
        with (output_dir / "main.c").open("a+") as main_file:
            main_file.write(
                pystache.render(
                    template,
                    {
                        "data_type": self.data_type,
                        "read_input": self.read_ext_input,
                        "verbose": self.verbose,
                        "format": (
                            "%d"
                            if np.issubdtype(self.data_type_py, np.integer)
                            else "%a" if self.to_hex else "%9g"
                        ),
                    },
                ),
            )

    def generate_makefile(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate Makefile build script."""
        header_files = []
        source_files = []
        for filename in self.files_to_gen:
            if ".c" in filename:
                source_files.append(filename)
            elif ".h" in filename:
                header_files.append(filename)

        # Configure Makefile template
        template = TemplateMakefile(
            self.function_name,
            compiler="gcc",
            compiler_flags=["-g", "-w", "-lm"],
            header_files=header_files,
            source_files=source_files,
        )
        if self.target_cfg is not None and "cflags" in self.target_cfg:
            template.add_compiler_flags(self.target_cfg["cflags"])
        # Generate Makefile
        with (output_dir / "Makefile").open("a+") as makefile:
            makefile.write(pystache.render(template))

    def generate_c_files(
        self: Self,
        c_files_directory: str | Path,
    ) -> None:
        """Generate C code implementation of current graph."""
        # Prepare output directory
        c_files_directory = Path(c_files_directory)
        c_files_directory.mkdir(exist_ok=True, parents=True)

        for file in self.files_to_gen:
            if (c_files_directory / file).exists():
                raise FileExistsError(c_files_directory / file)

        self.generate_function_source_file(c_files_directory)
        logging.info("Generated function source file.")
        self.generate_function_header_file(c_files_directory)
        logging.info("Generated function header file.")
        self.generate_globalvars_file(c_files_directory)
        logging.info("Generated globalvars .c file.")
        self.generate_main_file(c_files_directory)
        logging.info("Generated main file.")
        self.generate_makefile(c_files_directory)
        logging.info("Generated Makefile.")
        self.generate_test_dataset_files(c_files_directory)
        logging.info("Generated test_dataset files.")
        if self.target != "generic":
            self.generate_target_file(c_files_directory)
            logging.info("Generated target file.")
            self.generate_target_header_file(c_files_directory)
            logging.info("Generated target header file.")

    def generate_target_file(self: Self, output_dir: Path) -> None:
        logging.info("Generation of target file")
        mustach_hash = {}
        # mustach_hash[self.target] = True
        # Generate C code
        template = (
            Path(self.template_path) / self.target / "template_target_file.c.tpl"
        ).read_text()
        with (output_dir / "target.c").open("a+") as source_file:
            source_file.write(pystache.render(template, mustach_hash))

    def generate_target_header_file(self: Self, output_dir: Path) -> None:
        logging.info("Generation of target header file")
        mustach_hash = {}
        # mustach_hash[self.target] = True
        # Generate C code
        template = (
            Path(self.template_path) / self.target / "template_target_file.h.tpl"
        ).read_text()
        with (output_dir / "target.h").open("a+") as source_file:
            source_file.write(pystache.render(template, mustach_hash))

    def generate_function_source_file(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C Code for inference function."""
        mustach_hash = {
            "data_type": self.data_type,
            "input_size": self.layers[0].size,
            "output_size": self.layers[-1].size,
        }

        if self.target != "generic":
            mustach_hash["target_specific"] = True

        # Tag Gather-type layers
        if any(isinstance(i, Gather | GatherElements) for i in self.layers):
            gather_layers = [
                i for i in self.layers if isinstance(i, Gather | GatherElements)
            ]
            mustach_hash["is_gather"] = True
            indices = []
            for gather in gather_layers:
                indices.append(
                    {
                        "idx": f"{gather.idx:02d}",
                        "length": len(gather.indices.flatten()),
                        "list": self.flatten_array_order_c(gather.indices),
                    },
                )
            mustach_hash["indices"] = indices

        self.l_size_max = max((i.size for i in self.layers), default=1)

        if any(isinstance(i, Pooling2D | Conv2D | Gemm) for i in self.layers):
            mustach_hash["p"] = True

        if any(
            isinstance(
                i,
                Conv2D6loops | Conv2DStdGemm | Conv2DGemmTarget | Pooling2D | Gemm,
            )
            for i in self.layers
        ):
            mustach_hash["hw"] = True

        if any(isinstance(i, Dense | MatMul) for i in self.layers):
            mustach_hash["is_dense"] = True

        if any(
            isinstance(i, Conv2D6loops | AveragePooling2D | Softmax)
            for i in self.layers
        ):
            mustach_hash["is_sum"] = True

        if any(isinstance(i, MaxPooling2D) for i in self.layers):
            mustach_hash["is_max"] = True

        if any(isinstance(i, AveragePooling2D) for i in self.layers):
            mustach_hash["is_count"] = True

        if any(
            isinstance(i, ResizeLinear | ResizeCubic | ResizeNearest)
            for i in self.layers
        ):
            mustach_hash["is_resize"] = True

        if any(isinstance(layer, ResizeCubic) for layer in self.layers):
            mustach_hash["is_cubic_interpolation"] = True

        if any(isinstance(layer, ResizeLinear) for layer in self.layers):
            mustach_hash["is_linear_interpolation"] = True

        if any(isinstance(layer, Reduce) for layer in self.layers):
            mustach_hash["is_reduced"] = True

        if self.debug_mode in ["onnx", "keras"]:
            mustach_hash["debug_file"] = output_dir / "debug_file.txt"

        if self.debug_mode in ["time"]:
            mustach_hash["time"] = True

        mustach_hash["temp_data_type"] = (
            self.target_cfg["quantization"]["temp_dtype"]
            if self.target_cfg is not None
            else self.data_type
        )

        # Generate parameters per layer
        mustach_hash["layers"] = []
        for layer in self.layers:
            layer_hash = {
                "inference_function": layer.generate_inference_code_layer(),
                "path": layer.path,
                "size": layer.size,
            }

            if self.dict_cst and layer.idx in self.dict_cst:
                layer_hash["cst"] = True
                layer_hash["cst_name"] = self.dict_cst[layer.idx]

            if self.debug_mode in ["onnx", "keras"] and layer.idx in self.debug_target:
                layer_hash["debug_layer"] = True
                layer_hash["name"] = layer.name
                layer_hash["idx"] = layer.idx
                layer_hash["to_transpose"] = 0
                if (self.data_format == "channels_last") and (
                    hasattr(layer, "output_channels")
                ):
                    layer_hash["to_transpose"] = 1
                    layer_hash["channels"] = layer.output_channels
                    layer_hash["height"] = layer.output_height
                    layer_hash["width"] = layer.output_width

            if self.debug_mode in ["time"]:
                layer_hash["time"] = True
                layer_hash["name"] = layer.name
                layer_hash["idx"] = layer.idx

            mustach_hash["layers"].append(layer_hash)

        # Generate code to output graph data
        output_hash = {"path": self.layers[-1].path}
        if (self.data_format == "channels_last") and (
            hasattr(self.layers[-1], "output_channels")
        ):
            output_hash["output_channels"] = self.layers[-1].output_channels
            output_hash["output_height"] = self.layers[-1].output_height
            output_hash["output_width"] = self.layers[-1].output_width

            template = (
                Path(self.template_path)
                / "memory_layout/template_channels_last_output.c.tpl"
            ).read_text()
            mustach_hash["output_str"] = pystache.render(template, output_hash)
        else:
            output_hash["output_size"] = self.layers[-1].size
            if self.data_format == "channels_first":
                output_hash["comment"] = (
                    "Returning the output in channels first (ACETONE compute the result in channels first)"
                )
            else:
                output_hash["comment"] = "Returning the output (output flatten)"

            template = (
                Path(self.template_path)
                / "memory_layout/template_channels_first_output.c.tpl"
            ).read_text()
            mustach_hash["output_str"] = pystache.render(template, output_hash)

        if self.normalize:
            mustach_hash["pre_processing"] = self.Normalizer.write_pre_processing()
            mustach_hash["post_processing"] = self.Normalizer.write_post_processing()

        # Generate C code
        template = (Path(self.template_path) / "template_source_file.c.tpl").read_text()
        with (output_dir / "inference.c").open("a+") as source_file:
            source_file.write(pystache.render(template, mustach_hash))

    def generate_function_header_file(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C Code for graph structure."""
        mustach_hash = {
            "data_type": self.data_type,
            "path": list(range(self.maxpath)),
        }
        mustach_hash["temp_data_type"] = (
            self.target_cfg["quantization"]["temp_dtype"]
            if self.target_cfg is not None
            else self.data_type
        )

        self.nb_weights_max = 1
        self.nb_biases_max = 1

        self.patches_size_max = 1
        self.concate_size_max = 0
        for layer in self.layers:
            if (
                isinstance(layer, Conv2DStdGemm | Conv2DGemmTarget)
                and layer.patches_size > self.patches_size_max
            ):
                self.patches_size_max = layer.patches_size
            if isinstance(layer, Concatenate):
                self.patches_size_max = max(self.patches_size_max, layer.size)

        if any(
            isinstance(layer, Conv2DStdGemm | Conv2DGemmTarget) for layer in self.layers
        ):
            mustach_hash["path_size"] = max(self.l_size_max, self.patches_size_max)
        else:
            mustach_hash["path_size"] = self.l_size_max

        mustach_hash["cst"] = []
        written = {}
        for idx in self.dict_cst:
            for l in self.layers:
                if l.idx == idx:
                    layer = l
                    break

            if self.dict_cst[idx] not in written:
                written[self.dict_cst[idx]] = layer.size
            else:
                written[self.dict_cst[idx]] = max(
                    written[self.dict_cst[idx]],
                    layer.size,
                )

        for cst, s in written.items():
            mustach_hash["cst"].append({"name": cst, "size": s})

        # FIXME not all layers use the temp buffer but the list of layer types who do is unclear
        mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)

        # Collect layer parameters
        mustach_hash["layers"] = []
        for layer in self.layers:
            to_print = False
            layer_hash = {"name": layer.name, "idx": f"{layer.idx:02d}"}

            if hasattr(layer, "weights"):
                layer_hash["nb_weights"] = layer.nb_weights
                self.nb_weights_max = max(layer.nb_weights, self.nb_weights_max)
                to_print = True

            if isinstance(layer, ConstantLayer):
                layer_hash["nb_weights"] = layer.nb_weights
                to_print = True

            if hasattr(layer, "biases"):
                layer_hash["nb_biases"] = layer.nb_biases
                self.nb_biases_max = max(layer.nb_biases, self.nb_biases_max)
                to_print = True

            if isinstance(layer, Conv2DIndirectGemm):
                layer_hash["patches_size"] = layer.patches_size
                to_print = True

            if isinstance(layer, BatchNormalization):
                layer_hash["channels"] = layer.output_channels
                to_print = True

            if issubclass(type(layer), Broadcast) and layer.constant is not None:
                layer_hash["constant_size"] = layer.constant_size
                to_print = True

            if to_print:
                mustach_hash["layers"].append(layer_hash)

        if self.normalize:
            mustach_hash["normalization_cst"] = (
                self.Normalizer.write_normalization_cst_in_header_file()
            )

        # Generate header code
        template = (Path(self.template_path) / "template_header_file.c.tpl").read_text()
        with (output_dir / "inference.h").open("a+") as header_file:
            header_file.write(pystache.render(template, mustach_hash))

    def quantize_layers(self):
        if self.target_cfg is not None:
            for l in self.layers:
                try:
                    layer_qconf = self.target_cfg["quantization"]["layers"][
                        l.original_name
                    ]
                    if isinstance(l, ConstantLayer):
                        l.qparam = layer_qconf["out"]
                    else:
                        l.qparam = layer_qconf["params"]

                        # Add  activation
                        l.activation_function = QuantizeShiftActivation(
                            ctype=self.target_cfg["quantization"]["dtype"],
                            pytype=self.data_type_py,
                            qparam=layer_qconf["params"],
                            qin=layer_qconf["in"],
                            qout=layer_qconf["out"],
                            activation_function=l.activation_function,
                        )

                    (_, m) = qform.parse_q_format(l.qparam)
                    if hasattr(l, "weights"):
                        l.weights = np.rint(l.weights * (2**m - 1)).astype(
                            self.data_type_py,
                        )
                    if hasattr(l, "biases"):
                        l.biases = np.rint(l.biases * (2**m - 1)).astype(
                            self.data_type_py,
                        )
                    if hasattr(l, "constant") and l.constant is not None:
                        l.constant = np.rint(l.constant * (2**m - 1)).astype(
                            self.data_type_py,
                        )
                    l.qin = layer_qconf["in"]
                    l.qout = layer_qconf["out"]
                    logging.info(f"Quantize {l.name}_{l.idx} format {l.qparam}")
                    l.temp_pydtype = self.target_cfg["quantization"]["temp_pydtype"]
                    l.cdtype = self.target_cfg["quantization"]["dtype"]

                except KeyError:
                    if hasattr(l, "weights") or hasattr(l, "biases"):
                        raise KeyError(
                            f"Cannot quantize layer {l.name}_{l.idx}, missing data in target config",
                        )

    def generate_globalvars_file(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C Code for layer data."""
        mustach_hash = {
            "data_type": self.data_type,
            "path": list(range(self.maxpath)),
            "page_size": self.target_page_size,
        }

        if any(
            isinstance(layer, Conv2DStdGemm | Conv2DGemmTarget) for layer in self.layers
        ):
            mustach_hash["path_size"] = max(self.l_size_max, self.patches_size_max)
        else:
            mustach_hash["path_size"] = self.l_size_max

        mustach_hash["cst"] = []
        written = {}
        for idx in self.dict_cst:
            for l in self.layers:
                if l.idx == idx:
                    layer = l
                    break

            if self.dict_cst[idx] not in written:
                written[self.dict_cst[idx]] = layer.size
            else:
                written[self.dict_cst[idx]] = max(
                    written[self.dict_cst[idx]],
                    layer.size,
                )

        for cst in written:
            mustach_hash["cst"].append({"name": cst, "size": written[cst]})

        # FIXME not all layers use the temp buffer but the list of layer types who do is unclear
        mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)
        mustach_hash["temp_data_type"] = (
            self.target_cfg["quantization"]["temp_dtype"]
            if self.target_cfg is not None
            else self.data_type
        )

        if any(isinstance(layer, Conv2DIndirectGemm) for layer in self.layers):
            mustach_hash["zero"] = True

        mustach_hash["layers"] = []

        for layer in self.layers:
            to_print = False
            layer_hash = {"name": layer.name, "idx": f"{layer.idx:02d}"}

            if (w := getattr(layer, "weights", None)) is not None:
                layer_hash["nb_weights"] = layer.nb_weights
                layer_hash["weights"] = self.flatten_array_order_c(layer.weights)
                to_print = True

            if isinstance(layer, ConstantLayer):
                layer_hash["nb_weights"] = layer.constant.size
                layer_hash["weights"] = self.flatten_array_order_c(layer.constant)
                to_print = True

            if hasattr(layer, "biases"):
                layer_hash["nb_biases"] = layer.nb_biases
                layer_hash["biases"] = self.flatten_array_order_c(layer.biases)
                to_print = True

            if type(layer) is Conv2DIndirectGemm:
                layer_hash["patches_size"] = layer.patches_size
                layer_hash["patches"] = layer.create_ppatches()
                to_print = True

            if issubclass(type(layer), BatchNormalization):
                layer_hash["channels"] = layer.output_channels
                layer_hash["mean"] = self.flatten_array_order_c(layer.mean)
                layer_hash["var"] = self.flatten_array_order_c(layer.var)
                layer_hash["scale"] = self.flatten_array_order_c(layer.scale)
                to_print = True

            if issubclass(type(layer), Broadcast) and layer.constant is not None:
                layer_hash["constant"] = self.flatten_array_order_c(layer.constant)
                layer_hash["constant_size"] = layer.constant_size
                to_print = True

            if to_print:
                mustach_hash["layers"].append(layer_hash)

        template = Path(
            self.template_path + "template_global_var_file.c.tpl",
        ).read_text()
        with (output_dir / "global_vars.c").open("a+") as globalvars_file:
            globalvars_file.write(pystache.render(template, mustach_hash))
            if self.normalize:
                globalvars_file.write(
                    self.Normalizer.write_normalization_cst_in_globalvars_file(),
                )
