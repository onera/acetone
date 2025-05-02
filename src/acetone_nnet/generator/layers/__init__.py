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
    padding,
    pooling,
    reduce,
    resize,
gather,
gather_elements,
gemm,
input,
matmul,
softmax,
tile,
transpose
)
from .AddBias import AddBias
from .batch_normalization import BatchNormalization, BatchNormalizationDefault
from .broadcast import (
    Add,
    Average,
    Broadcast,
    Divide,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
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
from .Dot import Dot
from .flatten import Flatten, FlattenDefault
from .gather import Gather, GatherDefault
from .gather_elements import GatherElements, GatherElementsDefault
from .gemm import Gemm, GemmDefault
from .input import InputLayer, InputLayerDefault
from .matmul import MatMul, MatMulDefault
from .padding import ConstantPad, EdgePad, Pad, ReflectPad, WrapPad
from .pooling import AveragePooling2D, MaxPooling2D, Pooling2D
from .reduce import (
    Reduce,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
)
from .resize import Resize, ResizeCubic, ResizeLinear, ResizeNearest
from .softmax import SoftmaxDefault, Softmax
from .tile import Tile, TileDefault
from .transpose import Transpose, TransposeDefault

__all__ = (
    "AddBias",
    dense.__all__,
    "Dot",
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
