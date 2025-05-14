"""Instantiation of layers object for ONNX.

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
from typing import Any

import numpy as np
import onnx
from acetone_nnet.generator.activation_functions import (
    Clip,
    Exponential,
    LeakyReLu,
    Linear,
    Logarithm,
    ReLu,
    Sigmoid,
    TanH,
)
from acetone_nnet.generator.layers import (
    Add,
    AddBias,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    ConstantPad,
    Conv2D,
    Divide,
    EdgePad,
    Gather,
    GatherElements,
    Gemm,
    InputLayer,
    MatMul,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    ReflectPad,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Softmax,
    Subtract,
    Tile,
    Transpose,
    WrapPad,
)

###### Utility functions ######


def create_resize_obj(
        mode: bytes,
        **kwargs: object,
) -> ResizeCubic | ResizeLinear | ResizeNearest:
    """Create a resize layer."""
    if mode == b"nearest":
        return ResizeNearest(**kwargs)

    if mode == b"linear":
        return ResizeLinear(**kwargs)

    if mode == b"cubic":
        return ResizeCubic(**kwargs)

    raise ValueError("Error: resize mode " + mode.decode() + " not implemented")


def create_pad_obj(
        mode: bytes,
        **kwargs: object,
) -> ConstantPad | EdgePad | WrapPad | ReflectPad:
    """Create a Pad layer."""
    if mode == b"constant":
        return ConstantPad(**kwargs)

    if mode == b"edge":
        return EdgePad(**kwargs)

    if mode == b"wrap":
        return WrapPad(**kwargs)

    if mode == b"reflect":
        return ReflectPad(**kwargs)

    raise ValueError("Error: pad mode " + mode.decode() + " not implemented")


# Go find the constant named initializer_name in model(an onnx model)
def look_for_initializer(
        initializer_name: str,
        model: onnx.ModelProto,
) -> onnx.TensorProto | list:
    """Find the initializer named initializer_name."""
    if initializer_name == "":
        return []
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer
    return []


# Return a dictionary with {attribute: attribute_values}
def extract_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    """Return all attributes of an ONNX layer."""
    attributes = {}
    for attribute in node.attribute:
        attributes[attribute.name] = onnx.helper.get_attribute_value(attribute)
    return attributes


# Find the shape of shape_name in model (an onnx model)
# Depend on the characteristics of the model. Need value_info in the model
def get_shape(shape_name: str, model: onnx.ModelProto) -> list[int]:
    """Compute the shape of a tensor."""
    shape = []
    shape_length = 4
    for info in model.graph.value_info:
        if shape_name == info.name:
            shape = [info.type.tensor_type.shape.dim[i].dim_value
                     for i in range(len(info.type.tensor_type.shape.dim))]
    for input_layer in model.graph.input:
        if shape_name == input_layer.name:
            shape = [input_layer.type.tensor_type.shape.dim[i].dim_value for i in
                     range(len(input_layer.type.tensor_type.shape.dim))]
    for output in model.graph.output:
        if shape_name == output.name:
            shape = [output.type.tensor_type.shape.dim[i].dim_value for i in
                     range(len(output.type.tensor_type.shape.dim))]
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = 1
    if shape and len(shape) <= shape_length:
        shape = [1 for _i in range(4 - len(shape))] + shape
    return shape


# Return the size of the layer when given the list output_shape
def find_size(output_shape: list) -> int:
    """Compute the size of a tensor."""
    size = 1
    for i in output_shape:
        if i != 0:
            size *= i
    return size


###### Functions to create a Layer ######

# Create an input layers
def create_input_layer(
        input_layer: onnx.NodeProto,
        idx: int,
        dict_output: dict,
) -> InputLayer:
    """Create an Input layer."""
    dict_output[input_layer.name] = idx
    output_shape = [input_layer.type.tensor_type.shape.dim[i].dim_value for i in
                    range(len(input_layer.type.tensor_type.shape.dim))]
    size = find_size(output_shape)

    return InputLayer(idx, size, output_shape, "channels_first")


# Create a layer Softmax
def create_softmax(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Softmax:
    """Create a Softmax layer."""
    onnx_version_change_implementation = 13

    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if "axis" not in attributes:
        if model.opset_import[0].version < onnx_version_change_implementation:
            attributes["axis"] = 1
        else:
            attributes["axis"] = len(output_shape) - 1
    if attributes["axis"] < 0:
        attributes["axis"] = len(output_shape) + attributes["axis"]
    return Softmax(idx=idx,
                   size=size,
                   output_shape=output_shape,
                   axis=attributes["axis"])


# Create a layer Conv
def create_conv(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Conv2D:
    """Create a Conv2D layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name, model)
                    for initializer_name in node.input[1:]]
    attributes = extract_attributes(node)
    if "dilations" not in attributes:
        attributes["dilations"] = [1]
    if "group" not in attributes:
        attributes["group"] = 1
    if ("auto_pad" not in attributes) or (attributes["auto_pad"].decode() == "NOTSET"):
        if "pads" not in attributes:
            attributes["auto_pad"] = "VALID"
        else:
            attributes["auto_pad"] = attributes["pads"]
    else:
        attributes["auto_pad"] = attributes["auto_pad"].decode()
    if "strides" not in attributes:
        attributes["strides"] = [1, 1]

    initializers_length = 2
    if len(initializers) == initializers_length:
        biases = onnx.numpy_helper.to_array(initializers[1])
    else:
        biases = np.zeros(output_shape[1])

    return Conv2D(
        conv_algorithm="specs",
        idx=idx,
        size=size,
        padding=attributes["auto_pad"],
        strides=attributes["strides"][0],
        kernel_h=attributes["kernel_shape"][0],
        kernel_w=attributes["kernel_shape"][1],
        dilation_rate=attributes["dilations"][0],
        nb_filters=initializers[0].dims[0],
        input_shape=input_shape,
        output_shape=output_shape,
        weights=np.moveaxis(onnx.numpy_helper.to_array(initializers[0]), 0, 3),
        biases=biases,
        activation_function=Linear(),
    )


# Create a layer Concat
def create_concat(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Concatenate:
    """Create a Concatenate layer."""
    input_shapes = []
    for input_value in node.input:
        input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    return Concatenate(
        idx,
        size,
        attributes["axis"],
        input_shapes,
        output_shape,
        activation_function=Linear(),
    )


# Create a layer Resize
def resize_set_default_attributes_values(
        attributes: dict[str, Any],
        input_shape: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Set default value in attributes."""
    if "axes" not in attributes:
        attributes["axes"] = []
    else:
        for i in range(len(attributes["axes"])):
            axe = attributes["axes"][i]
            if axe < 0:
                axe = len(input_shape) - axe
    if "coordinate_transformation_mode" not in attributes:
        attributes["coordinate_transformation_mode"] = "half_pixel"
    if "exclude_outside" not in attributes:
        attributes["exclude_outside"] = 0
    if "keep_aspect_ratio_policy" not in attributes:
        attributes["keep_aspect_ratio_policy"] = "stretch"
    if "extrapolation_value" not in attributes:
        attributes["extrapolation_value"] = 0.0
    if "nearest_mode" not in attributes:
        attributes["nearest_mode"] = "round_prefer_floor"
    if "cubic_coeff_a" not in attributes:
        attributes["cubic_coeff_a"] = -0.75

    return attributes


def create_resize(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ResizeCubic | ResizeLinear | ResizeNearest:
    """Create a Resize layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name, model)
                    for initializer_name in node.input[1:]]
    attributes = extract_attributes(node)
    attributes = resize_set_default_attributes_values(attributes, input_shape)

    if initializers[2]:
        # target size
        targ_size = onnx.numpy_helper.to_array(initializers[2])
        bool_resize = 0
    else:
        # scale
        targ_size = onnx.numpy_helper.to_array(initializers[1])
        bool_resize = 1

    roi = onnx.numpy_helper.to_array(initializers[0]) if initializers[0] else []

    return create_resize_obj(
        mode=attributes["mode"],
        idx=idx,
        size=size,
        input_shape=input_shape,
        axes=attributes["axes"],
        coordinate_transformation_mode=attributes["coordinate_transformation_mode"],
        exclude_outside=attributes["exclude_outside"],
        keep_aspect_ratio_policy=attributes["keep_aspect_ratio_policy"],
        target_size=targ_size,
        boolean_resize=bool_resize,
        roi=roi,
        extrapolation_value=attributes["extrapolation_value"],
        nearest_mode=attributes["nearest_mode"],
        cubic_coeff_a=attributes["cubic_coeff_a"],
        activation_function=Linear(),
    )


# create a layer Pad
def create_pad(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ConstantPad | EdgePad | WrapPad | ReflectPad:
    """Create a Pad layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name, model)
                    for initializer_name in node.input[1:]]
    attributes = extract_attributes(node)

    initializer_length = 2
    if len(initializers) == initializer_length:
        axes = []
    else:
        axes = onnx.numpy_helper.to_array(initializers[2])
        for i in range(len(axes)):
            axe = axes[i]
            if axe < 0:
                axe = len(input_shape) - axe
            axes[i] = axe
    return create_pad_obj(
        mode=attributes["mode"],
        idx=idx,
        size=size,
        pads=onnx.numpy_helper.to_array(initializers[0]),
        constant_value=onnx.numpy_helper.to_array(initializers[1]),
        axes=axes,
        input_shape=input_shape,
        activation_function=Linear(),
    )


# create a layer Gather
def create_gather(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Gather:
    """Create a Gather layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    indices = onnx.numpy_helper.to_array(look_for_initializer(node.input[1], model))
    # Adjust indices to positive values from [-s, s-1] to [0, s]
    for i in np.ndindex(indices.shape):
        if indices[i] < 0:
            indices[i] = input_shape[attributes["axis"]] - abs(indices[i])
    return Gather(
        idx=idx,
        size=size,
        axis=attributes["axis"],
        indices=indices,
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


# create a layer Gather
def create_gather_elements(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> GatherElements:
    """Create a GatherElements layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    indices = onnx.numpy_helper.to_array(look_for_initializer(node.input[1], model))
    # Adjust indices to positive values from [-s, s-1] to [0, s]
    for i in np.ndindex(indices.shape):
        if indices[i] < 0:
            indices[i] = input_shape[attributes["axis"]] - abs(indices[i])
    return GatherElements(
        idx=idx,
        size=size,
        axis=attributes["axis"],
        indices=indices,
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


# create a layer Gemm
def create_gemm(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Gemm:
    """Create a Gemm layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    b_tensor = look_for_initializer(node.input[1], model)
    if len(node.input) == 3:  # noqa: PLR2004
        c_tensor = look_for_initializer(node.input[2], model)
    else:
        c_tensor = np.zeros((1, 1))
    if "transA" not in attributes:
        attributes["transA"] = 0
    if "transB" not in attributes:
        attributes["transB"] = 0
    if "alpha" not in attributes:
        attributes["alpha"] = 1.0
    if "beta" not in attributes:
        attributes["beta"] = 1.0
    return Gemm(
        idx=idx,
        size=size,
        alpha=attributes["alpha"],
        beta=attributes["beta"],
        transA=attributes["transA"],
        transB=attributes["transB"],
        weights=onnx.numpy_helper.to_array(b_tensor),
        bias=onnx.numpy_helper.to_array(c_tensor),
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


def matmul_compute_shape(
        shape: list,
) -> list:
    """Compute the shape of input/output tensor for MatMul(W,T)."""
    count_value = 3
    count = 0
    for i in shape:
        if i == 1:
            count += 1
    if count == count_value and shape[-1] != 1:
        shape = [1, 1, shape[-1], 1]
    return shape


def create_matmul(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> MatMul:
    """Create a MatMul layer."""
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    right_tensor = look_for_initializer(node.input[0], model)
    left_tensor = look_for_initializer(node.input[1], model)
    input_shape = []
    weights = []
    side = -1
    if left_tensor or right_tensor:
        if right_tensor and not left_tensor:
            # the weight is the right tensor:  MatMul(W,T)
            side = 1
            input_shape = get_shape(node.input[1], model)

            input_shape = matmul_compute_shape(input_shape)
            output_shape = matmul_compute_shape(output_shape)

            weights = onnx.numpy_helper.to_array(right_tensor)
            weights = np.reshape(weights, (input_shape[-2], 1, 1, output_shape[-2]))
            weights = np.moveaxis(weights, 0, 3)
            dict_input[idx] = [node.input[1]]
        if left_tensor and not right_tensor:
            # the weight is the right tensor:  MatMul(W,T)
            side = 0
            input_shape = get_shape(node.input[0], model)
            weights = onnx.numpy_helper.to_array(left_tensor)
            weights = np.reshape(weights, (output_shape[-1], 1, 1, input_shape[-1]))
            weights = np.moveaxis(weights, 0, 3)
            dict_input[idx] = [node.input[0]]
    else:
        dict_input[idx] = node.input
        input_shape = []
        for input_value in node.input:
            input_shape.append(get_shape(input_value, model))
        side = 2
        weights = None
        # to check
    return MatMul(idx=idx,
                  size=size,
                  input_shapes=input_shape,
                  weights=weights,
                  side=side,
                  activation_function=Linear())


def create_batch_norm(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> BatchNormalization:
    """Create a BatchNorm layer."""
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)

    if "epsilon" not in attributes:
        attributes["epsilon"] = 1e-05

    scale = look_for_initializer(node.input[1], model)
    biases = look_for_initializer(node.input[2], model)
    mean = look_for_initializer(node.input[3], model)
    var = look_for_initializer(node.input[4], model)

    return BatchNormalization(
        idx=idx,
        size=size,
        input_shape=output_shape,
        epsilon=attributes["epsilon"],
        scale=onnx.numpy_helper.to_array(scale),
        biases=onnx.numpy_helper.to_array(biases),
        mean=onnx.numpy_helper.to_array(mean),
        var=onnx.numpy_helper.to_array(var),
        activation_function=Linear(),
    )


def create_transpose(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Transpose:
    """Create a Transpose layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    return Transpose(
        idx=idx,
        size=size,
        input_shape=input_shape,
        perm=attributes["perm"],
        activation_function=Linear(),
    )


def create_tile(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Tile:
    """Create a Tile layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    repeats = onnx.numpy_helper.to_array(look_for_initializer(node.input[1], model))
    repeats = [int(rep) for rep in repeats]

    repeats_length = 4
    if len(repeats) < repeats_length:
        for _i in range(len(repeats), 4):
            repeats.insert(0, 1)
    return Tile(
        idx=idx,
        size=size,
        repeats=repeats,
        input_shape=input_shape,
        activation_function=Linear(),
    )


def create_reduce_sum(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ReduceSum:
    """Create a ReduceSum layer."""
    onnx_version_change_implementation = 13

    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if model.opset_import[0].version < onnx_version_change_implementation:
        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        attributes["noop_with_empty_axes"] = 0
        attributes["axes"] = onnx.numpy_helper.to_array(attributes["axes"])
    else:
        is_axes = 2

        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        if "noop_with_empty_axes" not in attributes:
            attributes["noop_with_empty_axes"] = 0
        if len(node.input) == is_axes:
            attributes["axes"] = onnx.numpy_helper.to_array(
                look_for_initializer(node.input[1], model),
            )
        else:
            attributes["axes"] = []

    return ReduceSum(
        idx=idx,
        size=size,
        axis=tuple(attributes["axes"]),
        keepdims=attributes["keepdims"],
        noop_with_empty_axes=attributes["noop_with_empty_axes"],
        input_shape=input_shape,
        activation_function=Linear(),
    )


def create_reduce_max(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ReduceMax:
    """Create a ReduceMax layer."""
    onnx_version_change_implementation = 18

    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if model.opset_import[0].version < onnx_version_change_implementation:
        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        attributes["noop_with_empty_axes"] = 0
        attributes["axes"] = onnx.numpy_helper.to_array(attributes["axes"])
    else:
        is_axes = 2

        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        if "noop_with_empty_axes" not in attributes:
            attributes["noop_with_empty_axes"] = 0
        if len(node.input) == is_axes:
            attributes["axes"] = onnx.numpy_helper.to_array(
                look_for_initializer(node.input[1], model),
            )
        else:
            attributes["axes"] = []

    return ReduceMax(
        idx=idx,
        size=size,
        axis=tuple(attributes["axes"]),
        keepdims=attributes["keepdims"],
        noop_with_empty_axes=attributes["noop_with_empty_axes"],
        input_shape=input_shape,
        activation_function=Linear(),
    )


def create_reduce_min(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ReduceMin:
    """Create a ReduceMin layer."""
    onnx_version_change_implementation = 18

    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if model.opset_import[0].version < onnx_version_change_implementation:
        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        attributes["noop_with_empty_axes"] = 0
        attributes["axes"] = onnx.numpy_helper.to_array(attributes["axes"])
    else:
        is_axes = 2

        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        if "noop_with_empty_axes" not in attributes:
            attributes["noop_with_empty_axes"] = 0
        if len(node.input) == is_axes:
            attributes["axes"] = onnx.numpy_helper.to_array(
                look_for_initializer(node.input[1], model),
            )
        else:
            attributes["axes"] = []

    return ReduceMin(
        idx=idx,
        size=size,
        axis=tuple(attributes["axes"]),
        keepdims=attributes["keepdims"],
        noop_with_empty_axes=attributes["noop_with_empty_axes"],
        input_shape=input_shape,
        activation_function=Linear(),
    )


def create_reduce_mean(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ReduceMean:
    """Create a ReduceMean layer."""
    onnx_version_change_implementation = 18

    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if model.opset_import[0].version < onnx_version_change_implementation:
        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        attributes["noop_with_empty_axes"] = 0
        attributes["axes"] = onnx.numpy_helper.to_array(attributes["axes"])
    else:
        is_axes = 2

        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        if "noop_with_empty_axes" not in attributes:
            attributes["noop_with_empty_axes"] = 0
        if len(node.input) == is_axes:
            attributes["axes"] = onnx.numpy_helper.to_array(
                look_for_initializer(node.input[1], model),
            )
        else:
            attributes["axes"] = []

    return ReduceMean(
        idx=idx,
        size=size,
        axis=tuple(attributes["axes"]),
        keepdims=attributes["keepdims"],
        noop_with_empty_axes=attributes["noop_with_empty_axes"],
        input_shape=input_shape,
        activation_function=Linear(),
    )


def create_reduce_prod(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> ReduceProd:
    """Create a ReduceReduceProd layer."""
    onnx_version_change_implementation = 18

    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input[0]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if model.opset_import[0].version < onnx_version_change_implementation:
        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        attributes["noop_with_empty_axes"] = 0
        attributes["axes"] = onnx.numpy_helper.to_array(attributes["axes"])
    else:
        is_axes = 2

        if "keepdims" not in attributes:
            attributes["keepdims"] = 1
        if "noop_with_empty_axes" not in attributes:
            attributes["noop_with_empty_axes"] = 0
        if len(node.input) == is_axes:
            attributes["axes"] = onnx.numpy_helper.to_array(
                look_for_initializer(node.input[1], model),
            )
        else:
            attributes["axes"] = []

    return ReduceProd(
        idx=idx,
        size=size,
        axis=tuple(attributes["axes"]),
        keepdims=attributes["keepdims"],
        noop_with_empty_axes=attributes["noop_with_empty_axes"],
        input_shape=input_shape,
        activation_function=Linear(),
    )


### Pooling layers ###

# Create a layer MaxPool
def create_max_pool(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> MaxPooling2D:
    """Create a MaxPool layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if "dilations" not in attributes:
        attributes["dilations"] = [1]
    if ("auto_pad" not in attributes) or (attributes["auto_pad"].decode() == "NOTSET"):
        if "pads" not in attributes:
            attributes["auto_pad"] = "VALID"
        else:
            attributes["auto_pad"] = attributes["pads"]
    else:
        attributes["auto_pad"] = attributes["auto_pad"].decode()
    if "strides" not in attributes:
        attributes["strides"] = [1, 1]
    return MaxPooling2D(
        idx=idx,
        size=size,
        padding=attributes["auto_pad"],
        strides=attributes["strides"][0],
        pool_size=attributes["kernel_shape"][0],
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


# create a layer AveragePool
def create_average_pool(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> AveragePooling2D:
    """Create an AveragePool layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributes = extract_attributes(node)
    if "dilations" not in attributes:
        attributes["dilations"] = [1]
    if ("auto_pad" not in attributes) or (attributes["auto_pad"].decode() == "NOTSET"):
        if "pads" not in attributes:
            attributes["auto_pad"] = "VALID"
        else:
            attributes["auto_pad"] = attributes["pads"]
    else:
        attributes["auto_pad"] = attributes["auto_pad"].decode()
    if "strides" not in attributes:
        attributes["strides"] = [1, 1]
    return AveragePooling2D(
        idx=idx,
        size=size,
        padding=attributes["auto_pad"],
        strides=attributes["strides"][0],
        pool_size=attributes["kernel_shape"][0],
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


# Create a layer GlobalAveragePool
def create_global_average_pool(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> AveragePooling2D:
    """Create a GlobalAveragePool layer."""
    input_shape = get_shape(node.input[0], model)
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return AveragePooling2D(
        idx=idx,
        size=size,
        padding=[0, 0, 0, 0],
        strides=1,
        pool_size=input_shape[2],
        input_shape=input_shape,
        output_shape=output_shape,
        activation_function=Linear(),
    )


### Broadcasts layers ###

# create a layer AddBias
##### UNUSED #####
def create_add_bias(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> AddBias:
    """Create an AddBias layer."""
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    right_tensor = look_for_initializer(node.input[0], model)
    left_tensor = look_for_initializer(node.input[1], model)
    biases = []
    if right_tensor:
        biases = onnx.numpy_helper.to_array(right_tensor)
        dict_input[idx] = [node.input[1]]
    elif left_tensor:
        biases = onnx.numpy_helper.to_array(left_tensor)
        dict_input[idx] = [node.input[0]]

    return AddBias(
        idx=idx,
        size=size,
        biases=biases,
        activation_function=Linear(),
    )


# create a layer Add
def create_add(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Add:
    """Create an Add layer."""
    constant_length = 4

    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    dict_input[idx] = []
    constant = np.zeros(get_shape(node.input[0], model))
    input_shapes = []
    for input_value in node.input:
        cst = look_for_initializer(input_value, model)
        if cst:
            constant = constant + onnx.numpy_helper.to_array(cst)
        else:
            dict_input[idx].append(input_value)
            input_shapes.append(get_shape(input_value, model))
    if constant.any():
        if len(constant.shape) < constant_length:
            for _i in range(4 - len(constant.shape)):
                constant = np.expand_dims(constant, axis=0)
        input_shapes.append(constant.shape)
    else:
        constant = None
    input_shapes = np.array(input_shapes)
    return Add(
        idx=idx,
        size=size,
        input_shapes=input_shapes,
        output_shape=output_shape,
        activation_function=Linear(),
        constant=constant,
    )


# create a layer Div
def create_div(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Divide:
    """Create a Divide layer."""
    constant_length = 4

    input_shapes = []
    constant = np.ones(get_shape(node.input[0], model))
    dict_input[idx] = []
    for input_value in node.input:
        factor = look_for_initializer(input_value, model)
        if factor:
            constant = constant / onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input_value)
            input_shapes.append(get_shape(input_value, model))

    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    # FIXME It seems like a test is missing here on the values of constant
    if not constant.all() and len(constant.shape) < constant_length:
        for _i in range(4 - len(constant.shape)):
            constant = np.expand_dims(constant, axis=0)
    input_shapes.append(constant.shape)
    input_shapes = np.array(input_shapes)
    return Divide(
        idx=idx,
        size=size,
        input_shapes=input_shapes,
        output_shape=output_shape,
        activation_function=Linear(),
        constant=constant,
    )


# create a layer Mul
def create_mul(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Multiply:
    """Create Multiply layer."""
    constant_length = 4

    input_shapes = []
    constant = np.ones(get_shape(node.input[0], model))
    dict_input[idx] = []
    for input_value in node.input:
        factor = look_for_initializer(input_value, model)
        if factor:
            constant = constant * onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input_value)
            input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    if not constant.all() and len(constant.shape) < constant_length:
        for _i in range(4 - len(constant.shape)):
            constant = np.expand_dims(constant, axis=0)
    input_shapes.append(list(constant.shape))
    input_shapes = np.array(input_shapes)
    return Multiply(
        idx=idx,
        size=size,
        input_shapes=input_shapes,
        output_shape=output_shape,
        activation_function=Linear(),
        constant=constant,
    )


# create a layer Sub
def create_sub(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Subtract:
    """Create a Subtract layer."""
    constant_lenght = 4

    input_shapes = []
    constant = np.zeros(get_shape(node.input[0], model))
    dict_input[idx] = []
    for input_value in node.input:
        factor = look_for_initializer(input_value, model)
        if factor:
            constant = constant - onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input_value)
            input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    if constant.any() and len(constant.shape) < constant_lenght:
        for _i in range(4 - len(constant.shape)):
            constant = np.expand_dims(constant, axis=0)
    input_shapes.append(constant.shape)
    input_shapes = np.array(input_shapes)
    return Subtract(
        idx=idx,
        size=size,
        input_shapes=input_shapes,
        output_shape=output_shape,
        activation_function=Linear(),
        constant=constant,
    )


# create a layer Max
def create_max(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Maximum:
    """Create a Max layer."""
    input_shapes = []
    for input_value in node.input:
        input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Maximum(
        idx=idx,
        size=size,
        input_shapes=np.array(input_shapes),
        output_shape=output_shape,
        activation_function=Linear(),
    )


# create a layer Min
def create_min(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Minimum:
    """Create a Min layer."""
    input_shapes = []
    for input_value in node.input:
        input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Minimum(
        idx=idx,
        size=size,
        input_shapes=np.array(input_shapes),
        output_shape=output_shape,
        activation_function=Linear(),
    )


# create a layer Average
def create_avg(
        node: onnx.NodeProto,
        idx: int,
        dict_input: dict,
        dict_output: dict,
        model: onnx.ModelProto,
) -> Average:
    """Create an Average layer."""
    input_shapes = []
    for input_value in node.input:
        input_shapes.append(get_shape(input_value, model))
    output_shape = get_shape(node.output[0], model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Average(
        idx=idx,
        size=size,
        input_shapes=input_shapes,
        output_shape=output_shape,
        activation_function=Linear(),
    )


###### Dict of all the functions ######
layer_type = {"Softmax": create_softmax,
              "Conv": create_conv,
              "Resize": create_resize,
              "Pad": create_pad,
              "Concat": create_concat,
              "Gather": create_gather,
              "GatherElements": create_gather_elements,
              "Gemm": create_gemm,
              "MatMul": create_matmul,
              "Transpose": create_transpose,
              "Tile": create_tile,
              "ReduceSum": create_reduce_sum,
              "ReduceMax": create_reduce_max,
              "ReduceMin": create_reduce_min,
              "ReduceMean": create_reduce_mean,
              "ReduceProd": create_reduce_prod,
              "MaxPool": create_max_pool,
              "AveragePool": create_average_pool,
              "GlobalAveragePool": create_global_average_pool,
              "Add": create_add,
              "Mul": create_mul,
              "Div": create_div,
              "Sub": create_sub,
              "Max": create_max,
              "Min": create_min,
              "Mean": create_avg}


###### Function to deal with the 'non important' layers of the graph ######

# Do the operation: Dropout.input = Dropout.output
def bypass(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
) -> None:
    """Bypass the layer (output=input)."""
    dict_output[node.output[0]] = dict_output.pop(node.input[0])


def create_initializer(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
) -> None:
    """Change a constant layer into an initializer."""
    const = model.graph.initializer.add()
    const.name = node.output[0]
    const.data_type = node.attribute[0].t.data_type
    const.raw_data = node.attribute[0].t.raw_data


###### Dict of all the functions ######
unused_layers = {"Dropout": bypass,
                 "Constant": create_initializer,
                 "Unsqueeze": bypass,
                 "Reshape": bypass,
                 "LRN": bypass,
                 "Shape": bypass,
                 "Flatten": bypass}


###### Function to fuse to ONNX layers ######

def fuse_relu(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer ReLu with the prior layer."""
    layers[dict_output[node.input[0]]].activation_function = ReLu()
    bypass(node, dict_output, model)


def fuse_tanh(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer TanH with the prior layer."""
    layers[dict_output[node.input[0]]].activation_function = TanH()
    bypass(node, dict_output, model)


def fuse_sigmoid(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer Sigmoid with the prior layer."""
    layers[dict_output[node.input[0]]].activation_function = Sigmoid()
    bypass(node, dict_output, model)


def fuse_exp(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer Exp with the prior layer."""
    layers[dict_output[node.input[0]]].activation_function = Exponential()
    bypass(node, dict_output, model)


def fuse_log(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer Log with the prior layer."""
    layers[dict_output[node.input[0]]].activation_function = Logarithm()
    bypass(node, dict_output, model)


def fuse_clip(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer Clip with the prior layer."""
    mini, maxi = float("-inf"), float("inf")
    if node.input[1]:
        mini = onnx.numpy_helper.to_array(look_for_initializer(node.input[1], model))[0]
    if node.input[2]:
        maxi = onnx.numpy_helper.to_array(look_for_initializer(node.input[2], model))[0]
    layers[dict_output[node.input[0]]].activation_function = Clip(
        max_value=maxi,
        min_value=mini,
    )
    bypass(node, dict_output, model)


def fuse_leaky_relu(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer  Leaky ReLu with the prior layer."""
    attribute = extract_attributes(node)
    if "alpha" not in attribute:
        attribute["alpha"] = 0.01
    layers[dict_output[node.input[0]]].activation_function = LeakyReLu(
        alpha=attribute["alpha"],
    )
    bypass(node, dict_output, model)


# Fuse a BatchNormalization layer with the previous Conv2D layer
def fuse_batch_normalization(
        node: onnx.NodeProto,
        dict_output: dict,
        model: onnx.ModelProto,
        layers: list,
) -> None:
    """Fuse the activation layer ReLu with the prior layer."""
    attributes = extract_attributes(node)
    if "epsilon" not in attributes:
        attributes["epsilon"] = 1e-05

    scale = onnx.numpy_helper.to_array(look_for_initializer(node.input[1], model))
    bias = onnx.numpy_helper.to_array(look_for_initializer(node.input[2], model))
    mean = onnx.numpy_helper.to_array(look_for_initializer(node.input[3], model))
    var = onnx.numpy_helper.to_array(look_for_initializer(node.input[4], model))

    weights = layers[dict_output[node.input[0]]].weights
    biases = layers[dict_output[node.input[0]]].biases

    for z in range(len(weights[0, 0, 0, :])):
        alpha = scale[z] / np.sqrt(var[z] + attributes["epsilon"])
        b = bias[z] - (mean[z] * alpha)
        weights[:, :, :, z] = alpha * weights[:, :, :, z]
        biases[z] = alpha * biases[z] + b

    layers[dict_output[node.input[0]]].weights = weights
    layers[dict_output[node.input[0]]].biases = biases

    bypass(node, dict_output, model)


###### Dict of all the functions ######
activation_layers = {"Relu": fuse_relu,
                     "Tanh": fuse_tanh,
                     "Sigmoid": fuse_sigmoid,
                     "Clip": fuse_clip,
                     "Exp": fuse_exp,
                     "Log": fuse_log,
                     "LeakyRelu": fuse_leaky_relu}
