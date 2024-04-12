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

from AddBias import Add_Bias
from Concatenate import Concatenate
from Dense import Dense
from Dot import Dot
from Flatten import Flatten
from Gather import Gather
from Gemm import Gemm
from Input import InputLayer
from MatMul import MatMul
from Softmax import Softmax
_layers = {"Add_Bias", "Concatenate", "Dense", "Dot", "Flatten", "Gather", "Gemm", "InputLayer", "MatMul", "Softmax"}

from Broadcast_layers.Add import Add
from Broadcast_layers.Average import Average
from Broadcast_layers.Broadcast import Broadcast
from Broadcast_layers.Divide import Divide
from Broadcast_layers.Maximum import Maximum
from Broadcast_layers.Minimum import Minimum
from Broadcast_layers.Multiply import Multiply
from Broadcast_layers.Subtract import Subtract
_broadcast = {"Add", "Average", "Broadcast", "Divide", "Maximum", "Minimum", "Multiply", "Subtract"}

from Conv_layers.Conv2D import Conv2D
from Conv_layers.Conv2D_6loops import Conv2D_6loops
from Conv_layers.Conv2D_gemm import Conv2D_gemm
from Conv_layers.Conv2D_indirect_gemm import Conv2D_indirect_gemm
from Conv_layers.Conv2D_std_gemm import Conv2D_std_gemm
_conv = {"Conv2D", "Conv2D_6loops", "Conv2D_gemm", "Conv2D_indirect_gemm", "Conv2D_std_gemm"}

from Pad_layers.Pad import Pad
from Pad_layers.ConstantPad import Constant_Pad
from Pad_layers.EdgePad import Edge_pad
from Pad_layers.ReflectPad import Reflect_pad
from Pad_layers.WrapPad import Wrap_pad
_pad = {"Pad", "Edge_pad", "Constant_Pad", "Reflect_pad", "Wrap_pad"}

from Pooling_layers.Pooling2D import Pooling2D
from Pooling_layers.AveragePooling2D import  AveragePooling2D
from Pooling_layers.MaxPooling2D import MaxPooling2D
_pooling = {"Pooling2D", "AveragePooling2D", "MaxPooling2D"}

from Resize_layers.Resize import Resize
from Resize_layers.ResizeCubic import ResizeCubic
from Resize_layers.ResizeLinear import ResizeLinear
from Resize_layers.ResizeNearest import ResizeNearest
_resize = {"Resize", "ResizeCubic", "ResizeLinear", "ResizeNearest"}

__all__ = list(
    _layers,
    _broadcast,
    _conv,
    _pad,
    _pooling,
    _resize
)