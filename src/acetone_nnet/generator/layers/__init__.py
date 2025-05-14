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

from . import (
    batch_normalization,
    broadcast,
    concatenate,
    convolution,
    dense,
    flatten,
    gather,
    gather_elements,
    gemm,
    input,
    matmul,
    padding,
    pooling,
    reduce,
    resize,
    softmax,
    tile,
    transpose,
)
from .batch_normalization import BatchNormalization, BatchNormalizationDefault
from .broadcast import (
    Add,
    AddDefault,
    Average,
    AverageDefault,
    Broadcast,
    Divide,
    DivideDefault,
    Maximum,
    MaximumDefault,
    Minimum,
    MinimumDefault,
    Multiply,
    MultiplyDefault,
    Subtract,
    SubtractDefault,
)
from .concatenate import Concatenate, ConcatenateDefault
from .convolution import (
    Conv2D,
    Conv2D6loops,
    Conv2DGemm,
    Conv2DGemmTarget,
    Conv2DIndirectGemm,
    Conv2DStdGemm,
)
from .dense import Dense, DenseDefault
from .flatten import Flatten, FlattenDefault
from .gather import Gather, GatherDefault
from .gather_elements import GatherElements, GatherElementsDefault
from .gemm import Gemm, GemmDefault
from .input import InputLayer, InputLayerDefault
from .matmul import MatMul, MatMulDefault
from .padding import (
    ConstantPad,
    ConstantPadDefault,
    EdgePad,
    EdgePadDefault,
    Pad,
    ReflectPad,
    ReflectPadDefault,
    WrapPad,
    WrapPadDefault,
)
from .pooling import (
    AveragePooling2D,
    AveragePooling2DDefault,
    MaxPooling2D,
    MaxPooling2DDefault,
    Pooling2D,
)
from .reduce import (
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
)
from .resize import (
    Resize,
    ResizeCubic,
    ResizeCubicDefault,
    ResizeLinear,
    ResizeLinearDefault,
    ResizeNearest,
    ResizeNearestDefault,
)
from .softmax import Softmax, SoftmaxDefault
from .tile import Tile, TileDefault
from .transpose import Transpose, TransposeDefault

__all__ = (
    dense.__all__,
    flatten.__all__,
    gather.__all__,
    gemm.__all__,
    input.__all__,
    matmul.__all__,
    softmax.__all__,
    transpose.__all__,
    gather_elements.__all__,
    tile.__all__,
    batch_normalization.__all__,
    concatenate.__all__,
    broadcast.__all__,
    convolution.__all__,
    padding.__all__,
    pooling.__all__,
    reduce.__all__,
    resize.__all__,
)
