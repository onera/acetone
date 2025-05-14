"""Tile Layer type definition.

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

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


class Tile(Layer):
    """Tile layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            repeats: list,
            input_shape: list,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a Tile layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.name = "Tile"
        self.repeats = repeats
        self.output_height = repeats[2] * self.input_height
        self.output_width = repeats[3] * self.input_width
        self.output_channels = repeats[1] * self.input_channels
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Tile (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Tile (size must be int)"
            msg += "\n"
        if any(type(rep) is not int for rep in self.repeats):
            msg += "Error: repeat type in Tile (all must be int"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Tile (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Tile (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Tile (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += "Error: activation function type in Tile (activation function must be a sub-classe of acetone_nnet Activation Function)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += f"Error: size value in Transpose ({self.size}!={self.output_channels * self.output_height * self.output_width})"
            msg += "\n"
        if len(self.repeats) != 4:
            msg += f"Error: repeats shape in Tile ({len(self.repeats)} non compatible with input of rank 4)"
            msg += "\n"
        for rep in self.repeats:
            if rep <= 0:
                msg += f"Error: repeat value in Tile ({rep} <= 0)"
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
        input_array = input_array.reshape((self.input_channels, self.input_height, self.input_width))
        return np.tile(input_array, self.repeats[1:])
