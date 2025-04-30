"""GatherElements Layer type definition.

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


# extract a list of subtensor from a given tensor
# attribut: axis alongside of which the submatrix will be extracted
# (if the desired submatrix must have the height, width or channels of the parent tensor)
# input: a tensor
# output: a list of tensor
class GatherElements(Layer):
    """GatherElements layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            axis: int,
            indices: np.ndarray,
            input_shape: list,
            output_shape: list,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a GatherElements layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "GatherElements"
        self.indices = indices
        self.axis = axis
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_channels = output_shape[1]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in GatherElements (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in GatherElements (size must be int)"
            msg += "\n"
        if type(self.axis) is not int:
            msg += "Error: axis type in GatherElements (axis must be int)"
            msg += "\n"
        if type(self.indices) is not np.ndarray:
            msg += "Error: indices in GatherElements (indices must be an numpy array)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in GatherElements (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in GatherElements (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in GatherElements (must be int)"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in GatherElements (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in GatherElements (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in GatherElements (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in GatherElements "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (f"Error: size value in GatherElements "
                    f"({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        if axis not in [1, 2, 3]:
            msg += (f"Error: axis out of bound in GatherElements "
                    f"({axis} for tensor in 4 dimension with first dimension unused)")
            msg += "\n"
        for indice in self.indices.flatten():
            if indice < 0 or indice >= input_shape[self.axis]:
                msg += (f"Error: indice out of bound in GatherElements "
                        f"({indice} out of bound for input of size {input_shape[self.axis]})")
                msg += "\n"
        if self.indices.shape[1:] != (self.output_channels, self.output_height, self.output_width):
            msg += (f"Error: non consistency between the indice shape and the output shape in GatherElements "
                    f"({self.indices.shape[1:]}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
            msg += "\n"
        if self.axis != 1 and self.output_channels != self.input_channels:
            msg += (f"Error: non consistency between the input shape and the output shape in GatherElements "
                    f"({input_shape}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
            msg += "\n"
        if self.axis != 2 and self.output_height != self.input_height:
            msg += (f"Error: non consistency between the input shape and the output shape in GatherElements "
                    f"({input_shape}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
            msg += "\n"
        if self.axis != 3 and self.output_width != self.input_width:
            msg += (f"Error: non consistency between the input shape and the output shape in GatherElements "
                    f"({input_shape}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
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
        output = np.zeros((self.output_channels, self.output_height, self.output_width))
        for f in range(self.output_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    if self.axis == 1:
                        output[f, i, j] = input_array[self.indices[0, f, i, j], i, j]
                    elif self.axis == 2:
                        output[f, i, j] = input_array[f, self.indices[0, f, i, j], j]
                    elif self.axis == 3:
                        output[f, i, j] = input_array[f, i, self.indices[0, f, i, j]]
        return output
