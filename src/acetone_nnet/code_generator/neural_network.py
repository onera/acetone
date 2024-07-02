"""*******************************************************************************
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
******************************************************************************
"""

import os
from abc import ABC
from pathlib import Path

import numpy as np
import onnx
import pystache
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential

from .. import templates
from ..format_importer.parser import parser
from .layers import (
    AveragePooling2D,
    BatchNormalization,
    Broadcast,
    Concatenate,
    Conv2D,
    Conv2D_6loops,
    Conv2D_indirect_gemm,
    Conv2D_std_gemm,
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


class CodeGenerator(ABC):

    def __init__(
        self,
        file: str | Path | onnx.ModelProto | Functional | Sequential,
        test_dataset: str | np.ndarray | None = None,
        function_name: str = "inference",
        nb_tests: int | str = 0,
        conv_algorithm: str = "std_gemm_nn",
        normalize: bool | str = False,
        debug_mode: str | None = None,
        **kwargs,
    ):

        self.file = file
        self.function_name = function_name
        self.nb_tests = int(nb_tests)
        self.conv_algorithm = conv_algorithm
        self.normalize = bool(normalize)
        self.template_path = templates.__file__[:-11]

        if not self.normalize:
            l, dtype, dtype_py, data_format, maxpath, dict_cst = parser(
                self.file,
                self.conv_algorithm,
            )
        elif self.normalize:
            l, dtype, dtype_py, data_format, maxpath, dict_cst, Normalizer = parser(
                self.file,
                self.conv_algorithm,
                normalize=self.normalize,
            )
            self.Normalizer = Normalizer

        self.layers = l
        self.data_type = dtype
        self.data_type_py = dtype_py
        self.maxpath = maxpath
        self.data_format = data_format
        self.dict_cst = dict_cst

        if type(test_dataset) == str:
            self.test_dataset_file = test_dataset
            ds = self.load_test_dataset()
            self.test_dataset = ds
        elif type(test_dataset) == np.ndarray:
            self.test_dataset = test_dataset
        else:
            print("creating random dataset")
            ds = self.create_test_dataset()
            self.test_dataset = ds

        self.files_to_gen = [
            "inference.c",
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
            raise TypeError(
                "Error: model type.\n Format must be: path to model, model ONNX or model Keras",
            )
        if not (
            type(test_dataset) == str
            or type(test_dataset) == np.ndarray
            or test_dataset == None
        ):
            raise TypeError(
                "Error: test_dataset type.\n Must be: path to dataset text filen, numpy array or None",
            )
        if not type(self.function_name) == str:
            raise TypeError("Error: function name type.\n Must be a string")
        if not type(self.conv_algorithm) == str:
            raise TypeError("Error: conv algorihtm type.\n Must be a string")
        if not (self.debug_mode == None or type(self.debug_mode) == str):
            raise TypeError("Error: debug mode type.\n Must be: string or None")
        if self.debug_mode:
            assert type(self.debug_target) == list

        ### Checking value consistency ###
        if self.conv_algorithm not in [
            "6loops",
            "indirect_gemm_nn",
            "indirect_gemm_tn",
            "indirect_gemm_nt",
            "indirect_gemm_tt",
            "std_gemm_nn",
            "std_gemm_tn",
            "std_gemm_nt",
            "std_gemm_tt",
        ]:
            raise ValueError(
                "Error: conv algorithm value.\n Must be one of: 6loops, indirect_gemm_nn, indirect_gemm_tn, indirect_gemm_nt, indirect_gemm_tt, std_gemm_nn, std_gemm_tn, std_gemm_nt, std_gemm_tt",
            )

        ##### Debug Mode #####
        if self.debug_mode:
            if self.debug_mode not in ["keras", "onnx"]:
                raise ValueError(
                    "Error: debug mode value.\n Must be one of: keras, onnx",
                )
        ##### Debug Mode #####

    def load_debug_target(self, debug_mode: str | None):
        targets = []
        for layer in self.layers[1:]:
            if debug_mode == "keras" and layer.name == "Softmax":
                targets[-1] = layer.idx
            else:
                targets.append(layer.idx)

        return targets

    def create_test_dataset(self):
        test_dataset = self.data_type_py(
            np.random.default_rng(seed=10).random(
                (self.nb_tests, 1, int(self.layers[0].size)),
            ),
        )
        return test_dataset

    def load_test_dataset(self):

        test_dataset = []
        try:
            with open(self.test_dataset_file) as f:
                for i, line in enumerate(f):
                    line = line[1:-2].split(",")
                    if self.data_type == "int":
                        line = list(map(int, line))
                    elif self.data_type == "double":
                        line = list(map(float, line))
                    elif self.data_type == "float":
                        line = list(map(np.float32, line))
                    test_dataset.append(line)
                    if i == self.nb_tests - 1:
                        break
            test_dataset = np.array(test_dataset)
            f.close()

        except TypeError:
            None

        return test_dataset

    def compute_inference(self, c_files_directory: str):
        with open(os.path.join(c_files_directory, "output_python.txt"), "w+") as fi:
            for nn_input in self.test_dataset:

                ##### Debug Mode #####
                if self.debug_mode:
                    debug_output = []
                    targets = []
                ##### Debug Mode #####

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
                previous_layer_result = [
                    nn_input for i in range(self.maxpath)
                ]  # for the very first layer, it is the neural network input

                to_store = {}  # a dictionnary containing the values to store
                for layer in self.layers:
                    if not layer.previous_layer:
                        previous_layer_result[layer.path] = layer.forward_path_layer(
                            previous_layer_result[layer.path],
                        )  # if the layer is an input layer, it directly take the vaue from it's path
                    elif len(layer.previous_layer) == 1:
                        if len(layer.previous_layer[0].next_layer) == 1:
                            previous_layer_result[layer.path] = (
                                layer.forward_path_layer(
                                    previous_layer_result[layer.previous_layer[0].path],
                                )
                            )  # if the layer has exactly one previous_layer, it takes the value from it's father's path
                        else:
                            previous_layer_result[layer.path] = (
                                layer.forward_path_layer(
                                    to_store[layer.previous_layer[0].idx],
                                )
                            )  # if the father is stored, we take it from the storage
                            layer.previous_layer[
                                0
                            ].sorted += (
                                1  # the number of children already "taken care of"
                            )
                    else:  # if the layer has multiple ancestors, we take all of their value
                        prev_layer = []
                        for prev in layer.previous_layer:
                            if len(prev.next_layer) == 1:
                                prev_layer.append(previous_layer_result[prev.path])
                            else:
                                prev_layer.append(to_store[prev.idx])
                                prev.sorted += 1
                        previous_layer_result[layer.path] = layer.forward_path_layer(
                            prev_layer,
                        )

                    # After having computed the value of the layer, we check if there is a fused layer.
                    if layer.fused_layer:
                        # If the current layer is the last layer to be updated, the fused layer must be computed
                        if (
                            layer.fused_layer.count_updated_prior_layers
                            == len(layer.fused_layer.prior_layers) - 1
                        ):
                            fused = layer.fused_layer
                            # The layer has multiple ancestors (otherwise, it is treated as an activation function)
                            prev_layer = []
                            for prev in fused.prior_layers:
                                if (
                                    prev.idx not in to_store
                                ):  # Taking the value where it is stored
                                    prev_layer.append(previous_layer_result[prev.path])
                                else:
                                    prev_layer.append(to_store[prev.idx])
                                    prev.sorted += 1
                            previous_layer_result[layer.path] = (
                                fused.forward_path_layer(prev_layer)
                            )
                        # Else, we notify the fused layer that another one of its ancestors have been computed
                        else:
                            layer.fused_layer.count_updated_prior_layers += 1

                    if (
                        len(layer.next_layer) > 1
                    ):  # if the layer has more than one child, it needs to be stored
                        to_store[layer.idx] = previous_layer_result[layer.path]

                    for prev in layer.previous_layer:
                        if (prev.sorted == len(prev.next_layer)) and (
                            prev in to_store
                        ):  # if all the children of the parent layer are "taken care of", we "forget" the parent's value ( *2 because of the creation of the dict in graph.to_save)
                            to_store.pop(prev.idx)

                    ##### Debug Mode #####
                    if self.debug_mode:
                        # Add the inference result of the layer to debug_output
                        if layer.name != "Input_layer":
                            if layer.idx in self.debug_target:
                                debug_output.append(previous_layer_result[layer.path])
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
                    ##### Debug Mode #####

                nn_output = previous_layer_result[layer.path]
                # print(nn_output) # write to file instead

                # Write results in text files to compare prediction.
                if (self.data_format == "channels_last") and hasattr(
                    layer,
                    "output_channels",
                ):
                    nn_output = np.transpose(nn_output, (1, 2, 0))
                nn_output = np.reshape(nn_output, -1)

                if self.normalize:
                    nn_output = self.Normalizer.post_processing(nn_output)

                for j in range(len(nn_output)):
                    print(f"{nn_output[j]:.9g}", end=" ", file=fi, flush=True)
                    # print(decimal.Decimal(nn_output[j]), end=' ', file=fi, flush=True)
                print(" ", file=fi)

        fi.close()

        print("File output_python.txt generated.")

        if self.debug_mode:
            return debug_output, targets
        else:
            return nn_output

    def flatten_array_orderc(self, array: np.ndarray):

        flattened_aray = array.flatten(order="C")
        s = "\n        {"
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i]) + ", "
        s = s[:-2]
        s += "}"

        return s

    def flatten_array(self, array: np.ndarray):
        s = "\n        {"
        shape = array.shape
        if len(shape) < 4:
            for i in range(4 - len(shape)):
                shape = (1,) + shape
            array = np.reshape(array, shape)
        for j in range(shape[3]):
            for k in range(shape[0]):
                for f in range(shape[1]):
                    for i in range(shape[2]):
                        s += str(array[k, f, i, j]) + ", "
        s = s[:-2]
        s += "}"
        return s

    def generate_testdataset_files(self, output_dir: Path):

        testdataset_header = open(output_dir / "test_dataset.h", "w+")
        testdataset_source = open(output_dir / "test_dataset.c", "w+")

        with open(
            self.template_path + "template_test_dataset_header.c.tpl",
        ) as template_file:
            template = template_file.read()
        template_file.close()

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
        testdataset_header.close()

        dataset = "{"
        if self.test_dataset is None:
            pass
        else:
            for j in range(self.test_dataset.shape[0]):
                dataset += self.flatten_array_orderc(self.test_dataset[j]) + ","
            dataset = dataset[:-1]

        dataset += "};\n"

        with open(
            self.template_path + "template_test_dataset_source.c.tpl",
        ) as template_file:
            template = template_file.read()
        template_file.close()

        testdataset_source.write(
            pystache.render(
                template,
                {"data_type": self.data_type, "dataset": dataset},
            ),
        )
        testdataset_source.close()

    def generate_main_file(self):

        with open(self.template_path + "template_main_file.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        self.main_file.write(pystache.render(template, {"data_type": self.data_type}))
        self.main_file.close()

    def generate_makefile(self):

        header_files = []
        source_files = []
        for filename in self.files_to_gen:
            if ".c" in filename:
                source_files.append(filename)
            elif ".h" in filename:
                header_files.append(filename)
            else:
                pass

        with open(self.template_path + "template_Makefile.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        self.makefile.write(
            pystache.render(
                template,
                {
                    "source_files": " ".join(source_files),
                    "header_files": " ".join(header_files),
                    "function_name": self.function_name,
                },
            ),
        )
        self.makefile.close()

    def generate_c_files(self, c_files_directory: str | Path):

        # Prepare output directory
        c_files_directory = Path(c_files_directory)
        if not c_files_directory.exists():
            c_files_directory.mkdir(exist_ok=True, parents=True)

        for file in self.files_to_gen:
            if (c_files_directory / file).exists():
                raise FileExistsError(c_files_directory / file)

        self.source_file = open(c_files_directory / "inference.c", "a+")
        self.header_file = open(c_files_directory / "inference.h", "a+")
        self.globalvars_file = open(c_files_directory / "global_vars.c", "a+")
        self.main_file = open(c_files_directory / "main.c", "a+")
        self.makefile = open(c_files_directory / "Makefile", "a+")

        self.generate_function_source_file(c_files_directory)
        print("Generated function source file.")
        self.generate_function_header_file()
        print("Generated function header file.")
        self.generate_globalvars_file()
        print("Generated globalvars .c file.")
        self.generate_main_file()
        print("Generated main file.")
        self.generate_makefile()
        print("Generated Makefile.")
        self.generate_testdataset_files(c_files_directory)
        print("Generated testdataset files.")

    def generate_function_source_file(self, output_dir: Path):

        mustach_hash = {}

        mustach_hash["data_type"] = self.data_type
        mustach_hash["input_size"] = self.layers[0].size
        mustach_hash["output_size"] = self.layers[-1].size

        if (
            len(
                gather_layers := [
                    i for i in self.layers if isinstance(i, Gather | GatherElements)
                ],
            )
            > 0
        ):
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

        self.l_size_max = 1
        if len(self.layers) > 0:
            self.l_size_max = max(i.size for i in self.layers)

        if any(
            (isinstance(layer, Dot | Pooling2D | Conv2D | Gemm))
            for layer in self.layers
        ):
            mustach_hash["p"] = True

        if any(
            (isinstance(layer, Conv2D_6loops | Conv2D_std_gemm | Pooling2D | Gemm))
            for layer in self.layers
        ):
            mustach_hash["hw"] = True

        if any(
            (isinstance(layer, Dense) or isinstance(layer, MatMul))
            for layer in self.layers
        ):
            mustach_hash["is_dense"] = True

        if (
            any(
                isinstance(layer, Conv2D_6loops)
                or isinstance(layer, AveragePooling2D)
                or isinstance(layer, Softmax),
            )
            for layer in self.layers
        ):
            mustach_hash["is_sum"] = True

        if any(isinstance(layer, MaxPooling2D) for layer in self.layers):
            mustach_hash["is_max"] = True

        if any(isinstance(layer, AveragePooling2D) for layer in self.layers):
            mustach_hash["is_count"] = True

        if any(
            isinstance(layer, ResizeLinear | ResizeCubic | ResizeNearest)
            for layer in self.layers
        ):
            mustach_hash["is_resize"] = True

        if any(isinstance(layer, ResizeCubic) for layer in self.layers):
            mustach_hash["is_cubic_interpolation"] = True

        if any(isinstance(layer, ResizeLinear) for layer in self.layers):
            mustach_hash["is_linear_interpolation"] = True

        if any(isinstance(layer, Reduce) for layer in self.layers):
            mustach_hash["is_reduced"] = True

        if self.debug_mode:
            mustach_hash["debug_file"] = output_dir / "debug_file.txt"

        mustach_hash["layers"] = []
        for layer in self.layers:
            layer_hash = {
                "inference_function": layer.generate_inference_code_layer(),
                "path": layer.path,
                "size": layer.size,
            }

            if self.dict_cst and layer in self.dict_cst:
                layer_hash["cst"] = True
                layer_hash["cst_name"] = self.dict_cst[layer]

            if self.debug_mode and layer.idx in self.debug_target:
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

            mustach_hash["layers"].append(layer_hash)

        output_hash = {"path": self.layers[-1].path}
        if (self.data_format == "channels_last") and (
            hasattr(self.layers[-1], "output_channels")
        ):
            output_hash["output_channels"] = self.layers[-1].output_channels
            output_hash["output_height"] = self.layers[-1].output_height
            output_hash["output_width"] = self.layers[-1].output_width

            with open(
                self.template_path
                + "memory_layout/template_channels_last_output.c.tpl",
            ) as template_file:
                template = template_file.read()
            template_file.close()
            mustach_hash["ouput_str"] = pystache.render(template, output_hash)

        else:
            output_hash["output_size"] = self.layers[-1].size
            if self.data_format == "channels_first":
                output_hash["comment"] = (
                    "Returning the output in channels first (ACETONE compute the result in channels first)"
                )
            else:
                output_hash["comment"] = "Returning the output (output flatten)"

            with open(
                self.template_path
                + "memory_layout/template_channels_first_output.c.tpl",
            ) as template_file:
                template = template_file.read()
            template_file.close()
            mustach_hash["ouput_str"] = pystache.render(template, output_hash)

        with open(self.template_path + "template_source_file.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        if self.normalize:
            mustach_hash["pre_processing"] = self.Normalizer.write_pre_processing()
            mustach_hash["post_processing"] = self.Normalizer.write_post_processing()

        self.source_file.write(pystache.render(template, mustach_hash))
        self.source_file.close()

    def generate_function_header_file(self):

        mustach_hash = {}
        mustach_hash["data_type"] = self.data_type
        mustach_hash["path"] = [i for i in range(self.maxpath)]

        self.nb_weights_max = 1
        self.nb_biases_max = 1

        self.patches_size_max = 1
        self.concate_size_max = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D_std_gemm):
                if layer.patches_size > self.patches_size_max:
                    self.patches_size_max = layer.patches_size
            if isinstance(layer, Concatenate):
                self.patches_size_max = max(self.patches_size_max, layer.size)

        if any(isinstance(layer, Conv2D_std_gemm) for layer in self.layers):
            mustach_hash["path_size"] = max(self.l_size_max, self.patches_size_max)
        else:
            mustach_hash["path_size"] = self.l_size_max

        mustach_hash["cst"] = []
        written = {}
        for layer in self.dict_cst:
            if self.dict_cst[layer] not in written:
                written[self.dict_cst[layer]] = layer.size
            else:
                written[self.dict_cst[layer]] = max(
                    written[self.dict_cst[layer]],
                    layer.size,
                )

        for cst in written:
            mustach_hash["cst"].append({"name": cst, "size": written[cst]})

        if (
            any(
                isinstance(layer, Concatenate)
                or any(isinstance(layer, Conv2D))
                or any(isinstance(layer, Dense))
                or any(issubclass(layer, Broadcast))
                or any(isinstance(layer, Gather))
                or any(isinstance(layer, Pad)),
            )
            for layer in self.layers
        ):
            mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)

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

            if type(layer) is Conv2D_indirect_gemm:
                layer_hash["patches_size"] = layer.patches_size
                to_print = True

            if type(layer) is BatchNormalization:
                layer_hash["channels"] = layer.output_channels
                to_print = True

            if issubclass(type(layer), Broadcast):
                if layer.constant is not None:
                    layer_hash["constant_size"] = layer.constant_size
                    to_print = True

            if to_print:
                mustach_hash["layers"].append(layer_hash)

        if self.normalize:
            mustach_hash["normalization_cst"] = (
                self.Normalizer.write_normalization_cst_in_header_file()
            )

        with open(self.template_path + "template_header_file.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        self.header_file.write(pystache.render(template, mustach_hash))
        self.header_file.close()

    def generate_globalvars_file(self):

        mustach_hash = {}

        mustach_hash["data_type"] = self.data_type
        mustach_hash["path"] = [i for i in range(self.maxpath)]

        if any(isinstance(layer, Conv2D_std_gemm) for layer in self.layers):
            mustach_hash["path_size"] = max(self.l_size_max, self.patches_size_max)
        else:
            mustach_hash["path_size"] = self.l_size_max

        mustach_hash["cst"] = []
        written = {}
        for layer in self.dict_cst:
            if self.dict_cst[layer] not in written:
                written[self.dict_cst[layer]] = layer.size
            else:
                written[self.dict_cst[layer]] = max(
                    written[self.dict_cst[layer]],
                    layer.size,
                )

        for cst in written:
            mustach_hash["cst"].append({"name": cst, "size": written[cst]})

        if (
            any(
                isinstance(layer, Concatenate)
                or any(isinstance(layer, Conv2D))
                or any(isinstance(layer, Dense))
                or any(issubclass(layer, Broadcast))
                or any(isinstance(layer, Gather))
                or any(isinstance(layer, Pad)),
            )
            for layer in self.layers
        ):
            mustach_hash["temp_size"] = max(self.l_size_max, self.patches_size_max)

        if any(isinstance(layer, Conv2D_indirect_gemm) for layer in self.layers):
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

            if type(layer) is Conv2D_indirect_gemm:
                layer_hash["patches_size"] = layer.patches_size
                layer_hash["patches"] = layer.create_ppatches()
                to_print = True

            if type(layer) is BatchNormalization:
                layer_hash["channels"] = layer.output_channels
                layer_hash["mean"] = self.flatten_array_orderc(layer.mean)
                layer_hash["var"] = self.flatten_array_orderc(layer.var)
                layer_hash["scale"] = self.flatten_array_orderc(layer.scale)
                to_print = True

            if issubclass(type(layer), Broadcast):
                if layer.constant is not None:
                    layer_hash["constant"] = self.flatten_array_orderc(layer.constant)
                    layer_hash["constant_size"] = layer.constant_size
                    to_print = True

            if to_print:
                mustach_hash["layers"].append(layer_hash)

        with open(
            self.template_path + "template_global_var_file.c.tpl",
        ) as template_file:
            template = template_file.read()
        template_file.close()

        self.globalvars_file.write(pystache.render(template, mustach_hash))

        if self.normalize:
            self.globalvars_file.write(
                self.Normalizer.write_normalization_cst_in_globalvars_file(),
            )

        self.globalvars_file.close()
