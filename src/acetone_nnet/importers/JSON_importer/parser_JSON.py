"""Parser to JSON files.

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
******************************************************************************
"""

import json
from itertools import islice
from pathlib import Path

import numpy as np

from acetone_nnet.generator import (
    ActivationFunctions,
    Add,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    ConstantPad,
    Conv2D,
    Dense,
    Flatten,
    InputLayer,
    LeakyReLu,
    Linear,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    ReLu,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Sigmoid,
    Softmax,
    Subtract,
    TanH,
)
from acetone_nnet.graph import graph_interpretor
from acetone_nnet.ir import Layer


def create_actv_function_obj(activation_str: str) -> ActivationFunctions:
    """Create an activation function."""
    if activation_str == "sigmoid":
        return Sigmoid()
    if activation_str == "relu":
        return ReLu()
    if activation_str == "tanh":
        return TanH()
    if activation_str == "linear":
        return Linear()
    if activation_str == "leaky_relu":
        return LeakyReLu(0.2)
    raise TypeError("Activation layer " + activation_str + " not implemented")


def create_resize_obj(
    mode: str,
    **kwargs: object,
) -> ResizeCubic | ResizeLinear | ResizeNearest:
    """Create a Resize layer."""
    if mode == "bicubic":
        return ResizeCubic(**kwargs)

    if mode == "bilinear":
        return ResizeLinear(**kwargs)

    if mode == "nearest":
        return ResizeNearest(**kwargs)

    raise ValueError("Error: resize mode " + mode + " not implemented")


def load_json(
    file_to_parse: str | Path,
) -> (list[Layer], str, type, str, int, dict[int, int]):
    """Load an JSON model and return the corresponding ACETONE representation."""
    file = Path.open(Path(file_to_parse))
    model = json.load(file)
    data_format = model["config"]["data_format"]

    data_type = model["config"]["layers"][0]["config"]["dtype"]

    if data_type == "float64":
        data_type = "double"
        data_type_py = np.float64

    elif data_type == "float32":
        data_type = "float"
        data_type_py = np.float32

    elif data_type == "int":
        data_type = "long int"
        data_type_py = np.int32

    layers = []

    l_temp = InputLayer(
        0,
        model["config"]["layers"][0]["config"]["size"],
        model["config"]["layers"][0]["config"]["input_shape"],
        data_format,
    )

    layers.append(l_temp)

    nb_softmax_layers = 0

    for idx, layer in list(islice(enumerate(model["config"]["layers"]), 1, None)):
        add_softmax_layer = False

        idx += nb_softmax_layers

        if (
            "activation" in layer["config"]
            and layer["config"]["activation"] == "softmax"
        ):
            layer["config"]["activation"] = "linear"
            add_softmax_layer = True

        if layer["class_name"] == "Dense":
            current_layer = Dense(
                original_name=layer["config"]["name"],
                idx=idx,
                size=layer["config"]["units"],
                weights=np.array(data_type_py(layer["weights"])),
                biases=data_type_py(layer["biases"]),
                activation_function=create_actv_function_obj(
                    layer["config"]["activation"],
                ),
            )

        elif layer["class_name"] == "Conv2D":
            current_layer = Conv2D(
                conv_algorithm="specs",
                original_name=layer["config"]["name"],
                idx=idx,
                size=layer["config"]["size"],
                padding=layer["config"]["padding"],
                strides=layer["config"]["strides"][0],
                kernel_h=layer["config"]["kernel_size"][0],
                kernel_w=layer["config"]["kernel_size"][1],
                dilation_rate=layer["config"]["dilation_rate"][0],
                nb_filters=layer["config"]["filters"],
                input_shape=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                weights=np.array(data_type_py(layer["weights"])),
                biases=data_type_py(layer["biases"]),
                activation_function=create_actv_function_obj(
                    layer["config"]["activation"],
                ),
            )

        elif layer["class_name"] == "AveragePooling2D":
            current_layer = AveragePooling2D(
                original_name=layer["config"]["name"],
                idx=idx,
                size=layer["config"]["size"],
                padding=layer["config"]["padding"],
                strides=layer["config"]["strides"][0],
                pool_size=layer["config"]["pool_size"][0],
                input_shape=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "MaxPooling2D":
            current_layer = MaxPooling2D(
                original_name=layer["config"]["name"],
                idx=idx,
                size=layer["config"]["size"],
                padding=layer["config"]["padding"],
                strides=layer["config"]["strides"][0],
                pool_size=layer["config"]["pool_size"][0],
                input_shape=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Flatten":
            current_layer = Flatten(
                original_name=layer["config"]["name"],
                idx=idx,
                size=layer["config"]["size"],
                input_shape=layer["config"]["input_shape"],
                data_format=data_format,
            )

        elif layer["class_name"] == "Add":
            current_layer = Add(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Multiply":
            current_layer = Multiply(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Subtract":
            current_layer = Subtract(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Concatenate":
            axis = layer["config"]["axis"]
            if data_format == "channels_last":
                axis = 1 if axis == 3 else axis + 1
            current_layer = Concatenate(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                axis=axis,
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Maximum":
            current_layer = Maximum(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Minimum":
            current_layer = Minimum(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "Average":
            current_layer = Average(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shapes=layer["config"]["input_shape"],
                output_shape=layer["config"]["output_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "ZeroPadding2D":
            pads = layer["config"]["padding"]
            if type(pads) is int:
                pads = [pads for i in range(8)]
            else:
                if type(pads[0]) is int:
                    pad_top, pad_bottom = pads[0], pads[0]
                    pad_left, pad_right = pads[1], pads[1]
                else:
                    pad_top, pad_bottom = pads[0][0], pads[0][1]
                    pad_left, pad_right = pads[1][0], pads[1][1]
                pads = [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]

            current_layer = ConstantPad(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                pads=pads,
                constant_value=0,
                axes=[],
                input_shape=layer["config"]["input_shape"],
                activation_function=Linear(),
            )

        elif layer["class_name"] == "BatchNormalization":
            BatchNormalization(
                original_name=layer["config"]["name"],
                idx=layer["config"]["idx"],
                size=layer["config"]["size"],
                input_shape=layer["config"]["input_shape"],
                epsilon=layer["config"]["epsilon"],
                scale=np.array(data_type_py(layer["gamma"])),
                biases=np.array(data_type_py(layer["beta"])),
                mean=np.array(data_type_py(layer["moving_mean"])),
                var=np.array(data_type_py(layer["moving_var"])),
                activation_function=Linear(),
            )

        elif (
            "Normalization" in layer["class_name"]
            or "Dropout" in layer["class_name"]
            or layer["class_name"] == "Reshape"
        ):
            continue

        else:
            raise TypeError("Error: layer" + layer["class_name"] + " not supported\n")

        for i in layer["config"]["prev_layer_idx"]:
            current_layer.previous_layer.append(layers[i])
            layers[i].next_layer.append(current_layer)

        l_temp = current_layer
        layers.append(current_layer)

        # Separated method to generate softmax
        if add_softmax_layer:
            nb_softmax_layers += 1
            current_layer = Softmax(
                original_name=f"Softmax_{idx+1}",
                idx=idx + 1,
                size=l_temp.size,
                output_shape=layer["config"]["output_shape"],
                axis=None,
            )
            l_temp.next_layer.append(current_layer)
            current_layer.previous_layer.append(l_temp)
            l_temp = current_layer
            layers.append(current_layer)

    layers, max_road, dict_cst = graph_interpretor.tri_topo(layers)
    layers = [x.find_output_str(dict_cst) for x in layers]
    print("Finished model initialization.")
    file.close()
    return layers, data_type, data_type_py, data_format, max_road, dict_cst
