"""Input Layer type definition.

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


class InputLayer(Layer):
    """Input layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shape: np.ndarray | list,
            data_format: str,
    ) -> None:
        """Build a Input layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shape = input_shape
        if len(self.input_shape) == 4:
            self.output_channels = self.input_shape[1]
            self.output_height = self.input_shape[2]
            self.output_width = self.input_shape[3]
        self.data_format = data_format
        self.name = "Input_layer"

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Input Layer (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Input Layer (size must be int)"
            msg += "\n"
        if any(type(shape) is not int for shape in self.input_shape[1:]):
            msg += "Error: input_shape in Input Layer (all dim must be int)"
            msg += "\n"
        if type(self.data_format) is not str:
            msg += "Error: data format type in Input Layer"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        prod = 1
        for shape in self.input_shape:
            if shape not in (None, 0):
                prod *= shape

        if self.size != prod:
            msg += f"Error: size value in Input Layer ({self.size}!={prod})"
            msg += "\n"
        if self.data_format not in ["channels_last", "channels_first"]:
            msg += f"Error: data format value in Input Layer ({self.data_format})"
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
        return input_array
