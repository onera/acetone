"""Parser to H5 files.

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

from itertools import islice

import keras
import numpy as np
from keras import activations
from keras.engine.base_layer import Layer
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential

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
    Sigmoid,
    Softmax,
    Subtract,
    TanH,
)
from acetone_nnet.graph import graph_interpretor


def get_layer_size(keras_layer: Layer) -> int:
    """Calculate the size of a layer."""
    size = 1
    if type(keras_layer.output_shape) is list:
        for dim in keras_layer.output_shape[0][1:]:
            size = size * dim
    else:
        for dim in keras_layer.output_shape[1:]:
            size = size * dim
    return size


def get_output_dimensions(
        output: list,
        data_format: str,
) -> list:
    """Get the shape of the layer's output."""
    dimensions = output[0] if type(output) is list else output

    if data_format == "channels_first":
        return dimensions
    if len(dimensions) == 4:
        return [dimensions[0], dimensions[3], dimensions[1], dimensions[2]]
    return dimensions


def get_input_dimensions(
        input_shape: list | np.ndarray,
        data_format: str,
) -> np.ndarray:
    """Get the shape of the layer's output."""
    dimensions = input_shape[0] if type(input_shape) is list and len(input_shape) == 1 \
        else input_shape

    if data_format == "channels_last":
        if type(dimensions) is list and len(dimensions[0]) == 4:
            dimensions = [(shape[0], shape[3], shape[1], shape[2])
                          for shape in dimensions]
        elif type(dimensions) is not list and len(dimensions) == 4:
            dimensions = (dimensions[0], dimensions[3], dimensions[1], dimensions[2])

    return np.array(dimensions)


def create_actv_function_obj(
        keras_activation_obj: activations,
) -> ActivationFunctions:
    """Create an activation function."""
    if keras_activation_obj == activations.sigmoid:
        return Sigmoid()
    if keras_activation_obj == activations.relu:
        return ReLu()
    if keras_activation_obj == activations.tanh:
        return TanH()
    if keras_activation_obj == activations.linear:
        return Linear()
    if keras_activation_obj == activations.leaky_relu:
        return LeakyReLu(0.2)
    if keras_activation_obj == activations.softmax:
        return Linear()

    msg = "Activation layer"
    raise TypeError(msg, keras_activation_obj.__name__, "not implemented")


def load_keras(
        file_to_parse: Functional | Sequential | str,
        debug: None | str,
) -> (list[Layer], str, type, str, int, dict[int, int]):
    """Load an H5 model and return the corresponding ACETONE representation."""
    if type(file_to_parse) is str:
        model = keras.models.load_model(file_to_parse)
    else:
        model = file_to_parse

    input_layer_size = 1
    for i in range(1, len(model.input.shape)):
        # start in idx 1 cause idx 0 represents batch size, so it's None in inference phase
        input_layer_size = input_layer_size * model.input.shape[i]

    data_format = "channels_first"
    if (hasattr(layer, "data_format") and layer.data_format == "channels_last" for layer in model.layers):
        data_format = "channels_last"

    data_type = model.layers[0].dtype

    if data_type == "float64":
        data_type = "double"
        data_type_py = np.float64

    elif data_type == "float32":
        data_type = "float"
        data_type_py = np.float32

    elif data_type == "int":
        data_type = "long int"
        data_type_py = np.int32

    else:
        raise TypeError("Type " + data_type + " not implemented")

    layers = []

    if model.layers[0].__class__.__name__ == "InputLayer":
        l_temp = InputLayer(idx=0,
                            size=get_layer_size(model.layers[0]),
                            input_shape=get_input_dimensions(model.layers[0].input_shape, data_format),
                            data_format=data_format)
        start = 1
    else:
        l_temp = InputLayer(idx=0,
                            size=input_layer_size,
                            input_shape=get_input_dimensions(model.input_shape, data_format),
                            data_format=data_format)
        start = 0

    layers.append(l_temp)

    nb_softmax_layers = 0

    for idx, layer_keras in list(islice(enumerate(model.layers), start, None)):
        add_softmax_layer = False
        idx += 1 - start
        idx += nb_softmax_layers

        if (hasattr(layer_keras, "activation")
                and layer_keras.activation == keras.activations.softmax):
            add_softmax_layer = True

        if layer_keras.__class__.__name__ == "Dense":
            weights = data_type_py(layer_keras.get_weights()[0])
            if len(weights.shape) < 3:
                for _i in range(3 - len(weights.shape)):
                    weights = np.expand_dims(weights, axis=-1)
            weights = np.moveaxis(weights, 2, 0)
            if len(weights.shape) < 4:
                weights = np.expand_dims(weights, axis=0)
            biases = data_type_py(layer_keras.get_weights()[1])
            current_layer = Dense(idx=idx,
                                  size=get_layer_size(layer_keras),
                                  weights=weights,
                                  biases=biases,
                                  activation_function=create_actv_function_obj(layer_keras.activation))

        elif layer_keras.__class__.__name__ == "Conv2D":
            weights = data_type_py(layer_keras.get_weights()[0])
            if len(weights.shape) < 3:
                for _i in range(3 - len(weights.shape)):
                    weights = np.expand_dims(weights, axis=-1)
            weights = np.moveaxis(weights, 2, 0)
            if len(weights.shape) < 4:
                weights = np.expand_dims(weights, axis=0)
            biases = data_type_py(layer_keras.get_weights()[1])
            current_layer = Conv2D(conv_algorithm="specs",
                                   idx=idx,
                                   size=get_layer_size(layer_keras),
                                   padding=layer_keras.padding,
                                   strides=layer_keras.strides[0],
                                   kernel_h=layer_keras.kernel_size[0],
                                   kernel_w=layer_keras.kernel_size[1],
                                   dilation_rate=layer_keras.dilation_rate[0],
                                   nb_filters=layer_keras.filters,
                                   input_shape=get_input_dimensions(layer_keras.input_shape, data_format),
                                   output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                   weights=weights,
                                   biases=biases,
                                   activation_function=create_actv_function_obj(layer_keras.activation))

        elif layer_keras.__class__.__name__ == "AveragePooling2D":
            current_layer = AveragePooling2D(idx=idx,
                                             size=get_layer_size(layer_keras),
                                             padding=layer_keras.padding,
                                             strides=layer_keras.strides[0],
                                             pool_size=layer_keras.pool_size[0],
                                             input_shape=get_input_dimensions(layer_keras.input_shape, data_format),
                                             output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                             activation_function=Linear())

        elif layer_keras.__class__.__name__ == "MaxPooling2D":
            current_layer = MaxPooling2D(idx=idx,
                                         size=get_layer_size(layer_keras),
                                         padding=layer_keras.padding,
                                         strides=layer_keras.strides[0],
                                         pool_size=layer_keras.pool_size[0],
                                         input_shape=get_input_dimensions(layer_keras.input_shape, data_format),
                                         output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                         activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Flatten":
            current_layer = Flatten(idx=idx,
                                    size=get_layer_size(layer_keras),
                                    input_shape=get_input_dimensions(layer_keras.input_shape, data_format),
                                    data_format=data_format)

        elif layer_keras.__class__.__name__ == "Add":
            current_layer = Add(idx=idx,
                                size=get_layer_size(layer_keras),
                                input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Multiply":
            current_layer = Multiply(idx=idx,
                                     size=get_layer_size(layer_keras),
                                     input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                     output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                     activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Subtract":
            current_layer = Subtract(idx=idx,
                                     size=get_layer_size(layer_keras),
                                     input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                     output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                     activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Maximum":
            current_layer = Maximum(idx=idx,
                                    size=get_layer_size(layer_keras),
                                    input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                    output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                    activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Minimum":
            current_layer = Minimum(idx=idx,
                                    size=get_layer_size(layer_keras),
                                    input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                    output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                    activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Average":
            current_layer = Average(idx=idx,
                                    size=get_layer_size(layer_keras),
                                    input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                    output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                    activation_function=Linear())

        elif layer_keras.__class__.__name__ == "Concatenate":
            axis = layer_keras.axis
            if data_format == "channels_last":
                axis = 1 if axis == 3 else axis + 1
            current_layer = Concatenate(idx=idx,
                                        size=get_layer_size(layer_keras),
                                        axis=axis,
                                        input_shapes=get_input_dimensions(layer_keras.input_shape, data_format),
                                        output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                        activation_function=Linear())

        elif layer_keras.__class__.__name__ == "ZeroPadding2D":
            pads = layer_keras.padding
            if type(pads) is int:
                pad_top, pad_bottom = pads, pads
                pad_left, pad_right = pads, pads
            elif type(pads[0]) is int:
                pad_top, pad_bottom = pads[0], pads[0]
                pad_left, pad_right = pads[1], pads[1]
            else:
                pad_top, pad_bottom = pads[0][0], pads[0][1]
                pad_left, pad_right = pads[1][0], pads[1][1]
            pads = [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]
            current_layer = ConstantPad(idx=idx,
                                        size=get_layer_size(layer_keras),
                                        pads=pads,
                                        constant_value=0,
                                        axes=[],
                                        input_shape=get_input_dimensions(layer_keras.input_shape, data_format),
                                        activation_function=Linear())

        elif layer_keras.__class__.__name__ == "BatchNormalization":
            if layers[-1].name == "Conv2D" and not debug:
                scale = data_type_py(layer_keras.get_weights()[0])
                bias = data_type_py(layer_keras.get_weights()[1])
                mean = data_type_py(layer_keras.get_weights()[2])
                var = data_type_py(layer_keras.get_weights()[3])

                weights = layers[-1].weights
                biases = layers[-1].biases

                for z in range(len(weights[0, 0, 0, :])):
                    alpha = scale[z] / np.sqrt(var[z] + layer_keras.epsilon)
                    B = bias[z] - (mean[z] * alpha)
                    weights[:, :, :, z] = alpha * weights[:, :, :, z]
                    biases[z] = alpha * biases[z] + B

                layers[-1].weights = weights
                layers[-1].biases = biases

                continue
            else:
                current_layer = BatchNormalization(idx=idx,
                                                   size=get_layer_size(layer_keras),
                                                   input_shape=get_input_dimensions(layer_keras.input_shape,
                                                                                    data_format),
                                                   epsilon=layer_keras.epsilon,
                                                   scale=data_type_py(layer_keras.get_weights()[0]),
                                                   biases=data_type_py(layer_keras.get_weights()[1]),
                                                   mean=data_type_py(layer_keras.get_weights()[2]),
                                                   var=data_type_py(layer_keras.get_weights()[3]),
                                                   activation_function=Linear())

        elif layer_keras.__class__.__name__ in ("Reshape", "Dropout"):
            continue

        else:
            raise TypeError("Error: layer " + layer_keras.__class__.__name__ + " not supported\n")

        if type(model) is keras.Sequential:
            if idx - 1 >= 0:
                current_layer.previous_layer.append(layers[idx - 1])
                layers[idx - 1].next_layer.append(current_layer)
        elif type(layer_keras.input) is list:
            for input in layer_keras.input:
                for k in range(len(model.layers)):
                    prev_layer = model.layers[k]
                    if prev_layer.name == input._keras_history.layer.name:
                        current_layer.previous_layer.append(layers[k])
                        layers[k].next_layer.append(current_layer)
                        break
        else:
            for k in range(len(model.layers)):
                prev_layer = model.layers[k]
                if prev_layer.name == layer_keras.input._keras_history.layer.name:
                    current_layer.previous_layer.append(layers[k])
                    layers[k].next_layer.append(current_layer)
                    break

        l_temp = current_layer
        layers.append(l_temp)

        if add_softmax_layer:
            nb_softmax_layers += 1
            current_layer = Softmax(idx=idx + 1,
                                    size=l_temp.size,
                                    output_shape=get_output_dimensions(layer_keras.output_shape, data_format),
                                    axis=None)
            l_temp.next_layer.append(current_layer)
            current_layer.previous_layer.append(l_temp)
            l_temp = current_layer
            layers.append(l_temp)

    layers, list_road, max_road, dict_cst = graph_interpretor.tri_topo(layers)
    layers = [x.find_output_str(dict_cst) for x in layers]
    print("Finished model initialization.")
    return layers, data_type, data_type_py, data_format, max_road, dict_cst
