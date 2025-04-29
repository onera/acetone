"""Flatten layer type definition.

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

from acetone_nnet.generator.Layer import Layer


class Flatten(Layer):
    """Flatten layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shape: list[int],
            data_format: str,
    ) -> None:
        """Build a Flatten layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shape = input_shape
        self.data_format = data_format
        self.name = "Flatten"

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if type(self.idx) is not int:
            msg = "Error: idx type in Flatten (idx must be int)"
            raise TypeError(msg)
        if type(self.size) is not int:
            msg = "Error: size type in Flatten (size must be int)"
            raise TypeError(msg)
        if any(type(shape) is not int for shape in self.input_shape[1:]):
            msg = "Error: input_shape in Flatten (all dim must be int)"
            raise TypeError(msg)
        if type(self.data_format) is not str:
            msg = "Error: data format type in Flatten"
            raise TypeError(msg)

        ### Checking value consistency ###
        if self.size != self.input_shape[1] * self.input_shape[2] * self.input_shape[3]:
            msg = (f"Error: size value in Flatten "
                   f"({self.size}!={self.input_shape[1] * self.input_shape[2] * self.input_shape[3]})")
            raise ValueError(msg)
        if self.data_format not in ["channels_last", "channels_first"]:
            msg = f"Error: data format value in Flatten ({self.data_format}"
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        if self.data_format == "channels_last":
            return np.transpose(np.reshape(input_array, self.input_shape[1:]), (1, 2, 0))
        return input_array
