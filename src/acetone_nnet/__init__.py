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

from . import debug, generator
from .cli.compare import cli_compare
from .generator import (
    ActivationFunctions,
    Add,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Broadcast,
    Clip,
    CodeGenerator,
    Concatenate,
    ConstantPad,
    Conv2D,
    Conv2D6loops,
    Conv2DGemm,
    Conv2DGemmTarget,
    Conv2DIndirectGemm,
    Conv2DStdGemm,
    Dense,
    Divide,
    EdgePad,
    Exponential,
    Flatten,
    Gather,
    GatherElements,
    Gemm,
    InputLayer,
    Layer,
    LeakyReLu,
    Linear,
    Logarithm,
    MatMul,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    Pad,
    Pooling2D,
    Reduce,
    ReduceMax,
    ReduceSum,
    ReflectPad,
    ReLu,
    Resize,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Sigmoid,
    Softmax,
    Subtract,
    TanH,
    Tile,
    Transpose,
    WrapPad,
)
from .versioning import layer_factories
from .versioning.versioning import list_all_implementations, register_factory

__all__ = (
    "cli_compare",
    "register_factory",
    "list_all_implementations",
    "layer_factories",
    generator.__all__,
    debug.__all__,
)
