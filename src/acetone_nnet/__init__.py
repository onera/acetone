"""
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

from . import code_generator
from .code_generator import (
    CodeGenerator, Layer,
    Add_Bias, Concatenate, Dense, Dot, Flatten, Gather, Gemm, InputLayer, MatMul, Softmax, BatchNormalization,
    Add, Average, Broadcast, Divide, Maximum, Minimum, Multiply, Subtract,
    Conv2D, Conv2D_6loops, Conv2D_gemm, Conv2D_indirect_gemm, Conv2D_std_gemm,
    Pad, Edge_pad, Wrap_pad, Reflect_pad, Constant_Pad,
    Pooling2D, MaxPooling2D, AveragePooling2D,
    Resize, ResizeCubic, ResizeLinear, ResizeNearest,
    ActivationFunctions, Linear, Sigmoid, ReLu, TanH, Exponential, Logarithm, Clip, LeakyReLu
)

from .cli_acetone import cli_acetone
from .cli_compare import cli_compare

__all__ = (
    "cli_acetone", "cli_compare",
    code_generator.__all__
)