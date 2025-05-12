"""Pooling layer base type definition.

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

import math

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


class Pooling2D(Layer):
    """Pooling layer base class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            padding: str | np.ndarray,
            strides: int,
            pool_size: int,
            input_shape: list[int],
            output_shape: list[int],
            activation_function: ActivationFunctions,
            **kwargs: int,
    ) -> None:
        """Instantiate a Pooling layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ""
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]

        self.output_channels = self.input_channels

        self.pooling_function = type(np.amax)
        self.local_var = ""
        self.local_var_2 = ""
        self.output_var = ""

        self.activation_function = activation_function

        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(
            self.padding,
            self.input_height,
            self.input_width,
            self.pool_size,
            self.pool_size,
            self.strides)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Pooling (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Pooling (size must be int)"
            msg += "\n"
        if type(self.padding) is not str and any(type(pad) is not int for pad in self.padding):
            msg += "Error: padding type in Pooling (must be str or ints)"
            msg += "\n"
        if type(self.strides) is not int:
            msg += "Error: strides type in Pooling (must be int)"
            msg += "\n"
        if type(self.pool_size) is not int:
            msg += "Error: pool_size type in Pooling (must be int)"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Pooling (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Pooling (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Pooling (must be int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in Pooling (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Pooling (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Pooling (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in Pooling "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += f"Error: size value in Pooling ({self.size}!={self.output_channels * self.output_height * self.output_width})"
            msg += "\n"
        if self.output_height != math.floor(
                (self.input_height + self.pad_bottom + self.pad_top - self.pool_size) / self.strides) + 1:
            msg += (f"Error: non consistency between the output height and the parameter of the operation in Pooling "
                    f"({self.output_height}!={math.floor((self.input_height + self.pad_bottom + self.pad_top - self.pool_size) / self.strides) + 1})")
            msg += "\n"
        if self.output_width != math.floor(
                (self.input_width + self.pad_left + self.pad_right - self.pool_size) / self.strides) + 1:
            msg += (f"Error: non consistency between the output width and the parameter of the operation in Pooling "
                    f"({self.output_width}!={math.floor((self.input_width + self.pad_left + self.pad_right - self.pool_size) / self.strides) + 1})")
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
        input_array = input_array.reshape(self.input_channels, self.input_height, self.input_width)

        output = np.zeros((self.input_channels, self.output_height, self.output_width))

        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom,
                                     self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input_array
        else:
            input_padded = input_array

        for c in range(self.input_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    output[c, i, j] = self.pooling_function(
                        input_padded[c, i * self.strides:i * self.strides + self.pool_size,
                        j * self.strides:j * self.strides + self.pool_size])

        return output
