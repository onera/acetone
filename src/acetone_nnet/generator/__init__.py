"""Init file.

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

from . import layers
from .activation_functions import (
    ActivationFunctions,
    Clip,
    Exponential,
    LeakyReLu,
    Linear,
    Logarithm,
    ReLu,
    Sigmoid,
    TanH,
)
from .Layer import Layer
from .layers import (
    Add,
    AddDefault,
    Average,
    AverageDefault,
    AveragePooling2D,
    AveragePooling2DDefault,
    BatchNormalization,
    BatchNormalizationDefault,
    Broadcast,
    Concatenate,
    ConcatenateDefault,
    ConstantPad,
    ConstantPadDefault,
    Conv2D,
    Conv2D6loops,
    Conv2DGemm,
    Conv2DGemmTarget,
    Conv2DIndirectGemm,
    Conv2DStdGemm,
    Dense,
    DenseDefault,
    Divide,
    DivideDefault,
    EdgePad,
    EdgePadDefault,
    Flatten,
    FlattenDefault,
    Gather,
    GatherDefault,
    GatherElements,
    GatherElementsDefault,
    Gemm,
    GemmDefault,
    InputLayer,
    InputLayerDefault,
    MatMul,
    MatMulDefault,
    Maximum,
    MaximumDefault,
    MaxPooling2D,
    MaxPooling2DDefault,
    Minimum,
    MinimumDefault,
    Multiply,
    MultiplyDefault,
    Pad,
    Pooling2D,
    Reduce,
    ReduceMax,
    ReduceMaxDefault,
    ReduceMean,
    ReduceMeanDefault,
    ReduceMin,
    ReduceMinDefault,
    ReduceProd,
    ReduceProdDefault,
    ReduceSum,
    ReduceSumDefault,
    ReflectPad,
    ReflectPadDefault,
    Resize,
    ResizeCubic,
    ResizeCubicDefault,
    ResizeLinear,
    ResizeLinearDefault,
    ResizeNearest,
    ResizeNearestDefault,
    Softmax,
    SoftmaxDefault,
    Subtract,
    SubtractDefault,
    Tile,
    TileDefault,
    Transpose,
    TransposeDefault,
    WrapPad,
    WrapPadDefault,
)
from .neural_network import CodeGenerator

__all__ = (
    "CodeGenerator",
    "ActivationFunctions",
    "Linear",
    "Sigmoid",
    "ReLu",
    "TanH",
    "Exponential",
    "Logarithm",
    "Clip",
    "LeakyReLu",
    "Layer",
    layers.__all__,
)
