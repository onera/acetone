from abc import ABC
from pathlib import Path
from typing import Any, Self

import numpy as np
import onnx
import pystache
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential

from acetone_nnet import templates
from acetone_nnet.generator.layers import (
    AveragePooling2D,
    BatchNormalization,
    Broadcast,
    Concatenate,
    Conv2D,
    Conv2D6loops,
    Conv2DGemmTarget,
    Conv2DIndirectGemm,
    Conv2DStdGemm,
    Dense,
    Dot,
    Gather,
    GatherElements,
    Gemm,
    MatMul,
    MaxPooling2D,
    Pad,
    Pooling2D,
    Reduce,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Softmax,
)
from acetone_nnet.importers.parser import parser
from acetone_nnet.templates.template_makefile import TemplateMakefile
from acetone_nnet.versioning.versioning import versioning


class CudaCodeGenerator(ABC):
    """Acetone CUDA Code Generator."""

    def __init__(
        self: Self,
        model_path: str | Path,
        test_dataset: str | np.ndarray | Path | None = None,
        function_name: str = "inference",
        nb_tests: int | str = 0,
        versions: dict[int, str] | dict[str, str] | None = None,
        normalize: bool | str = False,
        debug_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Configure the CUDA Code Generator.

        Parameters
        ----------
        model_path: str or Path
            Path to the model to generate the code for.
        test_dataset: ndarray or str or Path or None, optional
            Test dataset included in the generated code, either as an array of
            values or a path to an array. If None, a random dataset will be
            generated (see `nb_tests`).
        function_name: str, optional
            Name of the generated inference function.
        nb_tests: int or str, optional
            Number of generated test dataset entries if no default specified (see `test_dataset`)
        versions: dict or None, optional
            Dictionary capturing the specific implementation per layer or per
            layer type. If None, code generation will use the default implementation
            per layer type.
        normalize: bool or str, optional

        debug_mode: str or None, optional
        """
        self.model_path = model_path
        self.function_name = function_name
        self.nb_tests = int(nb_tests)
        self.normalize = bool(normalize)
        self.template_path = templates.__file__[:-11]

        if not self.normalize:
            l, dtype, dtype_py, data_format, maxpath, dict_cst = parser(
                file_to_parse=self.model_path,
            )
        else:  # normalise
            l, dtype, dtype_py, data_format, maxpath, dict_cst, normalizer = parser(
                file_to_parse=self.model_path,
                normalize=self.normalize,
            )
            self.Normalizer = normalizer

        self.layers: list[Any] = l
        self.versions = self._select_layers_implementation(versions)
        self.layers = versioning(self.layers, self.versions)

        self.data_type = dtype
        self.data_type_py = dtype_py
        self.maxpath = maxpath
        self.data_format = data_format
        self.dict_cst = dict_cst

        if isinstance(test_dataset, str | Path):
            self.test_dataset_file = test_dataset
            ds = self.load_test_dataset()
            self.test_dataset = ds
        elif isinstance(test_dataset, np.ndarray):
            self.test_dataset = test_dataset
        else:
            print("creating random dataset")
            ds = self.create_test_dataset()
            self.test_dataset = ds

        self.files_to_gen = [
            "inference.cpp",
            "inference.h",
            "global_vars.c",
            "main.c",
            "Makefile",
            "test_dataset.h",
            "test_dataset.c",
        ]

        ##### Debug Mode #####
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.debug_target = self.load_debug_target(debug_mode)
        ##### Debug Mode #####

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if not isinstance(
            self.file,
            str | Path | onnx.ModelProto | Functional | Sequential,
        ):
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

        ### Checking value consistency ###

        # Debug Mode
        if self.debug_mode and self.debug_mode not in ["keras", "onnx", "time"]:
            msg = "Error: debug mode value.\n Must be one of: keras, onnx, time"
            raise ValueError(msg)

    def _select_layers_implementation(
        self: Self,
        versions: dict[int, str] | dict[str, str] | None,
    ) -> dict[int, str]:
        """Create the dictionary used for the versioning.

        Parameters
        ----------
        versions: dict or None
            Name of selected implementation per layer, or per layer type. If
            None, the default implementation will be used for each layer type.

        """
        out_dict = {}
        if versions is None:
            for layer in self.layers:
                if layer.name == "Conv2D":
                    out_dict[layer.idx] = "std_gemm_nn"
        elif type(next(iter(versions))) is int:
            out_dict = versions
        else:
            for layer in self.layers:
                if layer.name in versions:
                    out_dict[layer.idx] = versions[layer.name]

        return out_dict

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

    def create_test_dataset(self: Self) -> np.ndarray:
        """Create random dataset for graph."""
        return np.random.default_rng(seed=10).random(
            size=(self.nb_tests, 1, int(self.layers[0].size)),
            dtype=self.data_type_py,
        )

    def load_test_dataset(self: Self) -> np.ndarray:
        """Load test dataset from file."""
        test_dataset: list[list[int | float | np.float32]] = []
        with Path(self.test_dataset_file).open() as f:
            for i, line in enumerate(f):
                contents = line[1:-2].split(",")
                if self.data_type == "int":
                    test_dataset.append(list(map(int, contents)))
                elif self.data_type == "double":
                    test_dataset.append(list(map(float, contents)))
                elif self.data_type == "float":
                    test_dataset.append(list(map(np.float32, contents)))
                if i >= self.nb_tests - 1:
                    break
        return np.array(test_dataset)

    def compute_inference(
        self: Self,
        c_files_directory: str,
    ) -> tuple[list, list] | np.ndarray:
        """Perform inference pass on test dataset."""
        with (Path(c_files_directory) / "output_python.txt").open("w+") as fi:
            for nn_input in self.test_dataset:
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
                    return inputs if len(inputs) > 1 else inputs[0]

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
                    if layer.fused_layer:
                        fused_inputs = gather_inputs(layer.fused_layer, layer_inputs)
                        layer_output = layer.fused_layer.forward_path_layer(
                            fused_inputs,
                        )

                    # Write Layer output into successors' input
                    for successor in layer.next_layer:
                        layer_inputs[successor.idx][layer.idx] = layer_output

                    # Cleanup layer inputs
                    del layer_inputs[layer.idx]

                    # Debug Mode
                    if self.debug_mode and layer.name != "Input_layer":
                        # Add the inference result of the layer to debug_output
                        if layer.idx in self.debug_target:
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

                for n in nn_output:
                    print(f"{n:.9g}", end=" ", file=fi, flush=True)
                print(" ", file=fi)

        print("File output_python.txt generated.")

        if self.debug_mode:
            return debug_output, targets
        return nn_output

    @staticmethod
    def flatten_array_orderc(array: np.ndarray) -> str:
        """Generate C flat array initializer in C order."""
        flattened_aray = array.flatten(order="C")
        s = "\n        {"
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i]) + ", "
        s = s[:-2]
        s += "}"

        return s

    @staticmethod
    def flatten_array(array: np.ndarray) -> str:
        """Generate C flat array initializer in Fortran order."""
        s = "\n        {"
        shape = array.shape
        if len(shape) < 4:
            for _i in range(4 - len(shape)):
                shape = (1, *shape)
            array = np.reshape(array, shape)
        for j in range(shape[3]):
            for k in range(shape[0]):
                for f in range(shape[1]):
                    for i in range(shape[2]):
                        s += str(array[k, f, i, j]) + ", "
        s = s[:-2]
        s += "}"
        return s

    def generate_testdataset_files(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C code for graph test dataset."""
        # Generate test header file
        template = (
            Path(self.template_path) / "template_test_dataset_header.c.tpl"
        ).read_text()
        with Path.open(output_dir / "test_dataset.h", "w+") as testdataset_header:
            testdataset_header.write(
                pystache.render(
                    template,
                    {
                        "nb_tests": self.nb_tests,
                        "nb_inputs": self.layers[0].size,
                        "nb_outputs": self.layers[-1].size,
                        "data_type": self.data_type,
                    },
                ),
            )

        # Generate test source file
        dataset = "{"
        if self.test_dataset is not None:
            dataset += ",".join(
                map(self.flatten_array_orderc, self.test_dataset),
            )
        dataset += "};\n"

        template = (
            Path(self.template_path) / "template_test_dataset_source.c.tpl"
        ).read_text()
        with Path.open(output_dir / "test_dataset.c", "w+") as testdataset_source:
            testdataset_source.write(
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
            main_file.write(pystache.render(template, {"data_type": self.data_type}))

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
        print("Generated function source file.")
        self.generate_function_header_file(c_files_directory)
        print("Generated function header file.")
        self.generate_globalvars_file(c_files_directory)
        print("Generated globalvars .c file.")
        self.generate_main_file(c_files_directory)
        print("Generated main file.")
        self.generate_makefile(c_files_directory)
        print("Generated Makefile.")
        self.generate_testdataset_files(c_files_directory)
        print("Generated testdataset files.")

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
                        "lenght": len(gather.indices.flatten()),
                        "list": self.flatten_array_orderc(gather.indices),
                    },
                )
            mustach_hash["indices"] = indices

        self.l_size_max = max((i.size for i in self.layers), default=1)

        if any(isinstance(i, Dot | Pooling2D | Conv2D | Gemm) for i in self.layers):
            mustach_hash["p"] = True

        if any(
            isinstance(
                i, Conv2D6loops | Conv2DStdGemm | Conv2DGemmTarget | Pooling2D | Gemm
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
            mustach_hash["ouput_str"] = pystache.render(template, output_hash)
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
            mustach_hash["ouput_str"] = pystache.render(template, output_hash)

        if self.normalize:
            mustach_hash["pre_processing"] = self.Normalizer.write_pre_processing()
            mustach_hash["post_processing"] = self.Normalizer.write_post_processing()

        # Generate C code
        template = (Path(self.template_path) / "template_source_file.c.tpl").read_text()
        with (output_dir / "inference.cpp").open("a+") as source_file:
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

        self.nb_weights_max = 1
        self.nb_biases_max = 1

        self.patches_size_max = 1
        self.concate_size_max = 0
        for layer in self.layers:
            if (
                isinstance(layer, Conv2DStdGemm | Conv2DGemmTarget)
            ) and layer.patches_size > self.patches_size_max:
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

        for cst in written:
            mustach_hash["cst"].append({"name": cst, "size": written[cst]})

        if any(
            isinstance(
                layer, Concatenate | Conv2D | Dense | Gather | Broadcast | Pad | Gemm
            )
            for layer in self.layers
        ):
            mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)

        # Collect layer parameters
        mustach_hash["layers"] = []
        for layer in self.layers:
            to_print = False
            layer_hash = {"name": layer.name, "idx": f"{layer.idx:02d}"}

            if hasattr(layer, "weights"):
                layer_hash["nb_weights"] = layer.nb_weights
                if layer.nb_weights > self.nb_weights_max:
                    self.nb_weights_max = layer.nb_weights
                to_print = True

            if hasattr(layer, "biases"):
                layer_hash["nb_biases"] = layer.nb_biases
                if layer.nb_biases > self.nb_biases_max:
                    self.nb_biases_max = layer.nb_biases
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

    def generate_globalvars_file(
        self: Self,
        output_dir: Path,
    ) -> None:
        """Generate C Code for layer data."""
        mustach_hash = {
            "data_type": self.data_type,
            "path": list(range(self.maxpath)),
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

        if any(
            isinstance(
                layer, Concatenate | Conv2D | Dense | Gather | Broadcast | Pad | Gemm
            )
            for layer in self.layers
        ):
            mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)

        if any(isinstance(layer, Conv2DIndirectGemm) for layer in self.layers):
            mustach_hash["zero"] = True

        mustach_hash["layers"] = []

        for layer in self.layers:
            to_print = False
            layer_hash = {"name": layer.name, "idx": f"{layer.idx:02d}"}

            if hasattr(layer, "weights"):
                layer_hash["nb_weights"] = layer.nb_weights
                layer_hash["weights"] = self.flatten_array(layer.weights)
                to_print = True

            if hasattr(layer, "biases"):
                layer_hash["nb_biases"] = layer.nb_biases
                layer_hash["biases"] = self.flatten_array_orderc(layer.biases)
                to_print = True

            if type(layer) is Conv2DIndirectGemm:
                layer_hash["patches_size"] = layer.patches_size
                layer_hash["patches"] = layer.create_ppatches()
                to_print = True

            if type(layer) is BatchNormalization:
                layer_hash["channels"] = layer.output_channels
                layer_hash["mean"] = self.flatten_array_orderc(layer.mean)
                layer_hash["var"] = self.flatten_array_orderc(layer.var)
                layer_hash["scale"] = self.flatten_array_orderc(layer.scale)
                to_print = True

            if issubclass(type(layer), Broadcast) and layer.constant is not None:
                layer_hash["constant"] = self.flatten_array_orderc(layer.constant)
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
