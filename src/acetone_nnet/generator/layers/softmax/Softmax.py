"""Softmax Layer type definition.

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
from typing_extensions import Self

from acetone_nnet.generator.Layer import Layer


class Softmax(Layer):
    """Softmax layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            output_shape: list,
            axis: int | None,
    ) -> None:
        """Build a Softmax layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "Softmax"
        self.axis = axis

        if self.size in output_shape:
            self.one_dimension = True
        else:
            self.output_channels = output_shape[1]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]
            self.one_dimension = False

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Softmax (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Softmax (size must be int)"
            msg += "\n"
        if type(self.axis) is not int and self.axis is not None:
            msg += "Error: axis type in Softmax (axis must be int or None)"
            msg += "\n"
        if not self.one_dimension:
            if type(self.output_channels) is not int:
                msg += "Error: output channels type in Softmax (must be int)"
                msg += "\n"
            if type(self.output_height) is not int:
                msg += "Error: output height type in Softmax (must be int)"
                msg += "\n"
            if type(self.output_width) is not int:
                msg += "Error: output width type in Softmax (must be int)"
                msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if not self.one_dimension and self.size != self.output_channels * self.output_height * self.output_width:
            msg += f"Error: size value in Transpose ({self.size}!={self.output_channels * self.output_height * self.output_width})"
            msg += "\n"
        if axis not in [1, 2, 3] and axis is not None:
            msg += f"Error: axis out of bound in Softmax ({axis} for tensor in 4 dimension with first dimension unused)"
            msg += "\n"
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        if not self.one_dimension:
            input_array = input_array.reshape(1, self.output_channels, self.output_height, self.output_width)
        else:
            input_array = input_array.flatten()
            self.axis = -1

        exp = np.exp(input_array, dtype=float)
        return exp / np.sum(exp, keepdims=True, axis=self.axis)
