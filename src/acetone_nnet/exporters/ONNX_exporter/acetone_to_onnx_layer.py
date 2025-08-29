"""Instantiation of ONNX layers object.

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

import numpy as np
import onnx
from onnx.helper import make_node, make_tensor_value_info, np_dtype_to_tensor_dtype
from onnxruntime_extensions import onnx_op

from acetone_nnet import MatMul
from acetone_nnet.generator.activation_functions import Linear
from acetone_nnet.generator.layers import (
    ActivationLayer,
    Add,
    Average,
    BatchNormalization,
    Concatenate,
    ConstantPad,
    Conv2D,
    Dense,
    Divide,
    EdgePad,
    Flatten,
    Gather,
    GatherElements,
    Gemm,
    InputLayer,
    Maximum,
    Minimum,
    Multiply,
    Pooling2D,
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
from acetone_nnet.ir import Layer

###### Utility functions ######

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
) -> onnx.TensorProto:
    """Create a TensorProto."""
    return onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )

def generate_input_output_name(
        layer: Layer,
) -> tuple[str, list[str]]:
    """Generate input and output name."""
    inputs_name = []
    for parent in layer.previous_layer:
        inputs_name.append(f"{parent.name}_{parent.idx}")
    if (
        hasattr(layer, "activation_function")
        and layer.activation_function is not None
        and not isinstance(layer.activation_function, Linear)
        and not isinstance(layer, ActivationLayer)
    ):
        output_name = f"{layer.name}_{layer.idx}_pre_activation"
    else:
        output_name = f"{layer.name}_{layer.idx}"

    return output_name, inputs_name


###### Functions to export a Layer ######

def export_batch_normalization(
        batch_normalization_layer: BatchNormalization,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE BatchNormalization layer to an ONNX BatchNormalization layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(batch_normalization_layer)
    scale_name = f"{batch_normalization_layer.name}_{batch_normalization_layer.idx}_scale"
    biases_name = f"{batch_normalization_layer.name}_{batch_normalization_layer.idx}_biases"
    mean_name = f"{batch_normalization_layer.name}_{batch_normalization_layer.idx}_mean"
    var_name = f"{batch_normalization_layer.name}_{batch_normalization_layer.idx}_var"
    inputs_name.extend([scale_name, biases_name, mean_name, var_name])

    node = make_node(
        name=batch_normalization_layer.original_name,
        op_type="BatchNormalization",
        inputs=inputs_name,
        outputs=[output_name],
        epsilon=batch_normalization_layer.epsilon,
    )

    scale = create_initializer_tensor(
        name=scale_name, tensor_array=batch_normalization_layer.scale, data_type=tensor_dtype,
    )
    biases = create_initializer_tensor(
        name=biases_name, tensor_array=batch_normalization_layer.biases, data_type=tensor_dtype,
    )
    mean = create_initializer_tensor(
        name=mean_name, tensor_array=batch_normalization_layer.mean, data_type=tensor_dtype,
    )
    var = create_initializer_tensor(
        name=var_name, tensor_array=batch_normalization_layer.var, data_type=tensor_dtype,
    )
    initializer = [scale, biases, mean, var]

    return node, initializer

def export_broadcast(
        broadcast_layer: Add | Average | Divide | Maximum | Minimum | Multiply | Subtract,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE broadcastable layer to an ONNX broadcastable layer (Add, Average, Divide, Maximum, Minimum, Multiply, Subtract)."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    op_type = broadcast_layer.name[:3]
    initializer = []

    output_name, inputs_name = generate_input_output_name(broadcast_layer)
    if broadcast_layer.constant is not None:
        constant_name = f"{broadcast_layer.name}_{broadcast_layer.idx}_constant"
        inputs_name.append(constant_name)

        constant = create_initializer_tensor(
            name=constant_name, tensor_array=np.squeeze(broadcast_layer.constant), data_type=tensor_dtype,
        )
        initializer.append(constant)
    elif len(inputs_name) == 1 and op_type in ["Add", "Sub", "Mul", "Div"]:
        constant_name = f"{broadcast_layer.name}_{broadcast_layer.idx}_constant"
        inputs_name.append(constant_name)
        constant_value = np.array(0) if op_type in ["Add", "Sub"] else np.array(1)
        constant = create_initializer_tensor(
            name=constant_name,
            tensor_array=constant_value,
            data_type=tensor_dtype,
        )
        initializer.append(constant)

    node = make_node(
        op_type, inputs_name, [output_name], name=broadcast_layer.original_name,
    )

    return node, initializer

def export_concatenate(
        concatenate_layer: Concatenate,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Concatenate layer to an ONNX Concatenate layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(concatenate_layer)

    node = make_node(
        "Concat",
        inputs_name,
        [output_name],
        axis=concatenate_layer.axis,
        name=concatenate_layer.original_name
    )

    return node, []

def export_conv2d(
        conv2d_layer: Conv2D,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Conv2D layer to an ONNX Conv2D layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(conv2d_layer)
    weight_name = f"{conv2d_layer.name}_{conv2d_layer.idx}_weight"
    bias_name = f"{conv2d_layer.name}_{conv2d_layer.idx}_bias"
    inputs_name.extend([weight_name,bias_name])
    node = make_node(
        name=conv2d_layer.original_name,
        op_type="Conv",
        inputs=inputs_name,
        outputs=[output_name],
        kernel_shape=(conv2d_layer.kernel_h, conv2d_layer.kernel_w),
        pads=(conv2d_layer.pad_right, conv2d_layer.pad_left, conv2d_layer.pad_bottom, conv2d_layer.pad_top),
        strides=(conv2d_layer.strides,conv2d_layer.strides),
        dilations=(conv2d_layer.dilation_rate, conv2d_layer.dilation_rate),
    )

    weights_array = np.moveaxis(conv2d_layer.weights, 3, 0)
    weights = create_initializer_tensor(
        name=weight_name, tensor_array=weights_array, data_type=tensor_dtype,
    )
    biases = create_initializer_tensor(
        name=bias_name, tensor_array=conv2d_layer.biases, data_type=tensor_dtype,
    )
    initializer = [weights, biases]

    return node, initializer


def export_dense(
        dense_layer: Dense,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE dense layer to an ONNX dense layer.

    The custom dense layer is made with onnxruntime_extension. (see: https://onnxruntime.ai/docs/extensions/add-op.html).
    """
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    @onnx_op(op_type="Dense",
             inputs=[tensor_dtype,tensor_dtype,tensor_dtype],
             outputs=[tensor_dtype])
    def dense(tensor:np.array,weight:np.array,bias:np.array) -> np.array:
        return np.dot(tensor,weight) + bias

    output_name, inputs_name = generate_input_output_name(dense_layer)
    weight_name = f"{dense_layer.name}_{dense_layer.idx}_weight"
    bias_name = f"{dense_layer.name}_{dense_layer.idx}_bias"
    inputs_name.extend([weight_name,bias_name])
    node = make_node(
        "Dense", inputs_name, [output_name], name=dense_layer.original_name
    )

    weights = create_initializer_tensor(
        name=weight_name, tensor_array=dense_layer.weights, data_type=tensor_dtype,
    )
    biases = create_initializer_tensor(
        name=bias_name, tensor_array=dense_layer.biases, data_type=tensor_dtype,
    )
    initializer = [weights, biases]

    return node, initializer

def export_flatten(
        flatten_layer: Flatten,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Flatten layer to an ONNX Flatten layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(flatten_layer)
    node = make_node(
        name=flatten_layer.original_name,
        op_type="Flatten",
        inputs=inputs_name,
        outputs=[output_name],
        axis=0,
    )

    return node, []

def export_gather(
        gather_layer: Gather,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Gather layer to an ONNX Gather layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(gather_layer)
    indices_name = f"{gather_layer.name}_{gather_layer.idx}_indices"
    inputs_name.append(indices_name)
    node = make_node(
        name=gather_layer.original_name,
        op_type="Gather",
        inputs=inputs_name,
        outputs=[output_name],
        axis=gather_layer.axis,
    )

    indices = create_initializer_tensor(
        name=indices_name,
        tensor_array=gather_layer.indices,
        data_type=tensor_dtype,
    )
    initializer = [indices]

    return node, initializer

def export_gather_elements(
        gather_elements_layer: GatherElements,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE GatherElements layer to an ONNX GatherElements layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(gather_elements_layer)
    indices_name = f"{gather_elements_layer.name}_{gather_elements_layer.idx}_indices"
    inputs_name.append(indices_name)
    node = make_node(
        name=gather_elements_layer.original_name,
        op_type="GatherElements",
        inputs=inputs_name,
        outputs=[output_name],
        axis=gather_elements_layer.axis,
    )

    indices = create_initializer_tensor(
        name=indices_name,
        tensor_array=gather_elements_layer.indices,
        data_type=tensor_dtype,
    )
    initializer = [indices]

    return node, initializer

def export_gemm(
    gemm_layer: Gemm,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Gemm layer to an ONNX Gemm layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(gemm_layer)
    weight_name = f"{gemm_layer.name}_{gemm_layer.idx}_weight"
    bias_name = f"{gemm_layer.name}_{gemm_layer.idx}_bias"
    inputs_name.extend([weight_name, bias_name])

    node = make_node(
        name=gemm_layer.original_name,
        op_type="Gemm",
        inputs=inputs_name,
        outputs=[output_name],
        alpha=1 if not gemm_layer.alpha else gemm_layer.alpha[0],
        beta=1 if not gemm_layer.beta else gemm_layer.beta[0],
        transA=gemm_layer.transpo[0],
        transB=gemm_layer.transpo[1],
    )

    weight = create_initializer_tensor(
        name=weight_name,
        tensor_array=gemm_layer.weights,
        data_type=tensor_dtype,
    )
    bias = create_initializer_tensor(
        name=bias_name,
        tensor_array=gemm_layer.biases,
        data_type=tensor_dtype,
    )
    initializer = [weight, bias]

    return node, initializer


def export_matmul(
        matmul_layer: MatMul,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE MatMul layer to an ONNX MatMul layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    initializer = []

    output_name, inputs_name = generate_input_output_name(matmul_layer)
    weight_name = f"{matmul_layer.name}_{matmul_layer.idx}_weight"
    if matmul_layer.side == 1:
        temp = [weight_name]
        temp.extend(inputs_name)
        inputs_name = temp
    elif matmul_layer.side == 0:
        inputs_name.append(weight_name)
    else:
        pass
    node = make_node(
        name=matmul_layer.original_name,
        op_type="MatMul",
        inputs=inputs_name,
        outputs=[output_name],
    )

    if matmul_layer.weights is not None:
        weight_value = matmul_layer.weights
        weight_value = np.reshape(weight_value, (weight_value.shape[-2],weight_value.shape[-1]))
        weight = create_initializer_tensor(
            name=weight_name,
            tensor_array=weight_value,
            data_type=tensor_dtype,
        )
        initializer.append(weight)

    return node, initializer

def export_pool(
        pool_layer: Pooling2D,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Pooling layer to an ONNX Pooling layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    op_type = {
        "AveragePooling2D": "AveragePool",
        "MaxPooling2D": "MaxPool",
    }

    output_name, inputs_name = generate_input_output_name(pool_layer)
    node = make_node(
        name=pool_layer.original_name,
        op_type=op_type[pool_layer.name],
        inputs=inputs_name,
        outputs=[output_name],
        strides=(pool_layer.strides, pool_layer.strides),
        kernel_shape=(pool_layer.pool_size, pool_layer.pool_size),
        pads=(pool_layer.pad_right, pool_layer.pad_left, pool_layer.pad_bottom, pool_layer.pad_top)
    )

    return node, []

def export_softmax(
        softmax_layer: Softmax,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE Softmax layer to an ONNX Softmax layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    output_name, inputs_name = generate_input_output_name(softmax_layer)
    node = make_node(
        name=softmax_layer.original_name,
        op_type="Softmax",
        inputs=inputs_name,
        outputs=[output_name],
    )

    return node, []

def export_pad(
        pad_layer: ConstantPad | EdgePad | ReflectPad | WrapPad,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE padding layer to an ONNX Pad layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(pad_layer)
    pads_name = f"{pad_layer.name}_{pad_layer.idx}_pads"
    axes_name = f"{pad_layer.name}_{pad_layer.idx}_axes" if pad_layer.axes else ""
    constant_name = f"{pad_layer.name}_{pad_layer.idx}_constant" if pad_layer.constant_value is not None else ""
    inputs_name.extend([pads_name, constant_name, axes_name])
    node = make_node(
        name=pad_layer.original_name,
        op_type="Pad",
        inputs=inputs_name,
        outputs=[output_name],
        mode=pad_layer.mode,
    )

    pads = create_initializer_tensor(
        name=pads_name,
        tensor_array=np.array(pad_layer.pads),
        data_type=onnx.TensorProto.INT64,
    )
    initializer = [pads]
    if pad_layer.axes :
        axes = create_initializer_tensor(
            name=axes_name,
            tensor_array=np.array(pad_layer.axes),
            data_type=onnx.TensorProto.INT64,
        )
        initializer.append(axes)
    if pad_layer.constant_value is not None:
        constant = create_initializer_tensor(
            name=constant_name,
            tensor_array=np.array(pad_layer.constant_value),
            data_type=tensor_dtype,
        )
        initializer.append(constant)

    return node, initializer

def export_resize(
        resize_layer: ResizeCubic | ResizeLinear | ResizeNearest,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export ACETONE resize layer to an ONNX Resize layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(resize_layer)
    scale_name = f"{resize_layer.name}_{resize_layer.idx}_scale"
    inputs_name.extend(["",scale_name,""])
    node = make_node(
        name=resize_layer.original_name,
        op_type="Resize",
        inputs=inputs_name,
        outputs=[output_name],
        axes=resize_layer.axes,
        coordinate_transformation_mode=resize_layer.coordinate_transformation_mode,
        cubic_coeff_a=resize_layer.cubic_coeff_a,
        exclude_outside=resize_layer.exclude_outside,
        extrapolation_value=resize_layer.extrapolation_value,
        keep_aspect_ratio_policy=resize_layer.keep_aspect_ratio_policy,
        mode=resize_layer.mode,
        nearest_mode=resize_layer.nearest_mode,
    )

    scale = create_initializer_tensor(
        name=scale_name,
        tensor_array=np.array(resize_layer.scale),
        data_type=tensor_dtype,
    )
    initializer = [scale]

    return node, initializer

def export_reduce(
        reduce_layer: ReduceMax | ReduceMean | ReduceMin | ReduceProd | ReduceSum,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE reduce layer to an ONNX reduce layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(reduce_layer)
    axes_name = f"{reduce_layer.name}_{reduce_layer.idx}_axes"
    inputs_name.append(axes_name)
    node = make_node(
        name=reduce_layer.original_name,
        op_type=reduce_layer.name,
        inputs=inputs_name,
        outputs=[output_name],
        keepdims=reduce_layer.keepdims,
        noop_with_empty_axes=reduce_layer.noop_with_empty_axes,
    )

    axes = create_initializer_tensor(
        name=axes_name,
        tensor_array=np.array(reduce_layer.axes),
        data_type=onnx.TensorProto.INT64,
    )
    initializer = [axes]

    return node, initializer

def export_tile(
        tile_layer: Tile,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Tile layer to an ONNX Tile layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(tile_layer)
    repeat_name = f"{tile_layer.name}_{tile_layer.idx}_repeat"
    inputs_name.append(repeat_name)
    node = make_node(
        name=tile_layer.original_name,
        op_type="Tile",
        inputs=inputs_name,
        outputs=[output_name],
    )

    repeat = create_initializer_tensor(
        name=repeat_name,
        tensor_array=np.array(tile_layer.repeats),
        data_type=tensor_dtype,
    )
    initializer = [repeat]

    return node, initializer

def export_transpose(
        transpose_layer: Transpose,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Transpose layer to an ONNX Transpose layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(transpose_layer)
    node = make_node(
        name=transpose_layer.original_name,
        op_type="Transpose",
        inputs=inputs_name,
        outputs=[output_name],
        perm=transpose_layer.perm,
    )

    return node, []

def get_activation_op_type(activation_layer: ActivationLayer) -> str | None:
    """Retrieve the corresponding ONNX op_type from an ACETONE Activation layer."""
    match activation_layer.name:
        case "Relu":
            return "Relu"
        case "Hyperb_tan":
            return "Tanh"
        case "Sigmoid":
            return "Sigmoid"
        case "Exponential":
            return "Exp"
        case "Logarithm":
            return "Log"
        case "Leakyrelu":
            return "LeakyRelu"
        case "Clip":
            return "Clip"
        case "Silu":
            return "Silu"
        case _:
            return None

def export_activation(
        activation_layer: ActivationLayer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Activation layer to an ONNX activation layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    output_name, inputs_name = generate_input_output_name(activation_layer)
    op_tye = get_activation_op_type(activation_layer)

    if op_tye == "Silu":
        @onnx_op(
            op_type="Silu",
            inputs=[tensor_dtype],
            outputs=[tensor_dtype],
        )
        def silu(tensor: np.array) -> np.array:
            return tensor / (1 + np.exp(-tensor))

    node = make_node(
        name=activation_layer.original_name,
        op_type=op_tye,
        inputs=inputs_name,
        outputs=[output_name],
    )
    return node, []

###### Dict of all the exporters ######
list_exporters = {
    "Add": export_broadcast,
    "Average": export_broadcast,
    "AveragePooling2D": export_pool,
    "BatchNormalization": export_batch_normalization,
    "Clip": export_activation,
    "Concatenate": export_concatenate,
    "ConstantPad": export_pad,
    "Conv2D": export_conv2d,
    "Dense": export_dense,
    "Divide": export_broadcast,
    "EdgePad": export_pad,
    "Exponential": export_activation,
    "Flatten": export_flatten,
    "Gather": export_gather,
    "GatherElements": export_gather_elements,
    "Gemm": export_gemm,
    "Hyperb_tan": export_activation,
    "Leakyrelu": export_activation,
    "Logarithm": export_activation,
    "MaxPooling2D": export_pool,
    "Maximum": export_broadcast,
    "MatMul": export_matmul,
    "Minimum": export_broadcast,
    "Multiply": export_broadcast,
    "ReflectPad": export_pad,
    "ReduceMax": export_reduce,
    "ReduceMean": export_reduce,
    "ReduceMin": export_reduce,
    "ReduceProd": export_reduce,
    "ReduceSum": export_reduce,
    "ResizeCubic": export_resize,
    "Relu": export_activation,
    "ResizeLinear": export_resize,
    "ResizeNearest": export_resize,
    "Sigmoid": export_activation,
    "Silu": export_activation,
    "Subtract": export_broadcast,
    "Softmax": export_softmax,
    "Tile": export_tile,
    "Transpose": export_transpose,
    "WrapPad": export_pad,
}


###### Functions to export activation functions ######

def export_relu(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE ReLu activation function to an ONNX ReLU layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Relu fonction from node {layer.original_name} ",
        op_type="Relu",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_silu(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Silu activation function to an ONNX makeshift silu layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))
    @onnx_op(
        op_type="Silu",
        inputs=[tensor_dtype],
        outputs=[tensor_dtype],
    )
    def silu(tensor: np.array) -> np.array:
        return tensor / (1 + np.exp(-tensor))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Silu fonction from node {layer.original_name} ",
        op_type="Silu",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_tanh(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Tanh activation function to an ONNX Tanh layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Tanh fonction from node {layer.original_name} ",
        op_type="Tanh",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_leaky_relu(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE LeakyRelu activation function to an ONNX LeakyRelu layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"LeakyRelu fonction from node {layer.original_name} ",
        op_type="LeakyRelu",
        inputs=[intput_name],
        outputs=[output_name],
        alpha=layer.activation_function.alpha,
    )

    return node, []

def export_sigmoid(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Sigmoid activation function to an ONNX Sigmoid layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Sigmoid fonction from node {layer.original_name} ",
        op_type="Sigmoid",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_exponential(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Exp activation function to an Exp ReLU layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Exp fonction from node {layer.original_name} ",
        op_type="Exp",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_log(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Log activation function to an Log ReLU layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"

    node = make_node(
        name=f"Log fonction from node {layer.original_name} ",
        op_type="Log",
        inputs=[intput_name],
        outputs=[output_name],
    )

    return node, []

def export_clip(
        layer: Layer,
        dtype_py:np.dtype,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    """Export an ACETONE Clip activation function to an Log Clip layer."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype_py.__name__))

    intput_name = f"{layer.name}_{layer.idx}_pre_activation"
    output_name= f"{layer.name}_{layer.idx}"
    min_name = f"{layer.name}_{layer.idx}_clip_min"
    max_name = f"{layer.name}_{layer.idx}_clip_max"

    node = make_node(
        name=f"Clip fonction from node {layer.original_name} ",
        op_type="Clip",
        inputs=[intput_name,min_name,max_name],
        outputs=[output_name],
    )

    min_initializer = create_initializer_tensor(
        name=min_name,
        tensor_array=np.array(layer.activation_function.min),
        data_type=tensor_dtype,
    )
    max_initializer = create_initializer_tensor(
        name=max_name,
        tensor_array=np.array(layer.activation_function.max),
        data_type=tensor_dtype,
    )
    initializer = [min_initializer, max_initializer]

    return node, initializer

###### Dict of all the activation exporters ######
list_activation_exporters = {
    "relu":export_relu,
    "hyperb_tan":export_tanh,
    "leakyrelu":export_leaky_relu,
    "sigmoid": export_sigmoid,
    "Exponential": export_exponential,
    "Logarithm": export_log,
    "Clip": export_clip,
    "silu": export_silu,
}
###### Functions to define the inputs and outputs of the graph ######
def export_input(
        input_layer: Layer,
        datatype_py:np.dtype,
) -> onnx.ValueInfoProto:
    """Export ACETONE Input layer to ONNX input value info."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(datatype_py.__name__))
    shape = getattr(input_layer,"input_shape", None)
    return make_tensor_value_info(
        name=f"{input_layer.name}_{input_layer.idx}",
        elem_type=tensor_dtype,
        shape=shape,
    )

def export_output(
        output_layer: Layer,
        datatype_py:np.dtype,
) -> onnx.ValueInfoProto:
    """Export ACETONE Input layer to ONNX input value info."""
    tensor_dtype = np_dtype_to_tensor_dtype(np.dtype(datatype_py.__name__))

    shape = [1,1,1,1]

    if isinstance(output_layer, Softmax) and output_layer.one_dimension:
        axis = -1 if output_layer.axis is None else output_layer.axis
        shape[axis] = output_layer.size
    elif isinstance(output_layer, InputLayer):
        shape = output_layer.input_shape
    else:
        if hasattr(output_layer, "output_channels"):
            shape[1] = output_layer.output_channels
        if hasattr(output_layer, "output_height"):
            shape[2]  = output_layer.output_height
        if hasattr(output_layer, "output_width"):
            shape[3]  = output_layer.output_width

    return make_tensor_value_info(
        name=f"{output_layer.name}_{output_layer.idx}",
        elem_type=tensor_dtype,
        shape=shape,
    )