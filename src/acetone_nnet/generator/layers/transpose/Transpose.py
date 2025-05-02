"""Transpose Layer type definition.

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
import pystache
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


class Transpose(Layer):
    """Transpose layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shape: list,
            perm: list,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a Transpose layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "Transpose"

        self.perm = perm

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.output_channels = input_shape[self.perm[1]]
        self.output_height = input_shape[self.perm[2]]
        self.output_width = input_shape[self.perm[3]]

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if not isinstance(self.idx, int):
            msg += "Error: idx type in Transpose (idx must be int)"
            msg += "\n"
        if not isinstance(self.size, int):
            msg += "Error: size type in Transpose (size must be int)"
            msg += "\n"
        if any(not isinstance(indice, int) for indice in self.perm):
            msg += "Error: perm type in Transpose (must be str or ints)"
            msg += "\n"
        if not isinstance(self.input_channels, int):
            msg += "Error: input channels type in Transpose (must be int)"
            msg += "\n"
        if not isinstance(self.input_height, int):
            msg += "Error: input height type in Transpose (must be int)"
            msg += "\n"
        if not isinstance(self.input_width, int):
            msg += "Error: input width type in Transpose (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in Transpose "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (f"Error: size value in Transpose "
                    f"({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        seen = []
        for indice in self.perm:
            if indice < 0 or indice >= 4:
                msg += (f"Error: perm out of bound in Transpose "
                        f"({indice} for tensor in 4 dimension with first dimension unused)")
                msg += "\n"
            if indice in seen:
                msg += f"Error: non unicity of perm's values in Transpose {self.perm}"
                msg += "\n"
            seen.append(indice)
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = np.reshape(
            input_array,
            (1, self.input_channels, self.input_height, self.input_width),
        )
        return np.transpose(input_array, self.perm)
