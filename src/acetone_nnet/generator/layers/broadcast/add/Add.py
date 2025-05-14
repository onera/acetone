"""Add layer type definition.

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


# Addition of several tensor
class Add(Layer):
    """Add layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shapes: list[np.ndarray],
            output_shape: list[int],
            activation_function: ActivationFunctions,
            constant: np.ndarray | float | None = None,
    ) -> None:
        """Instantiate an Add base layer."""
        Layer.__init__(self)
        self.idx = idx
        self.size = size
        self.name = "Add"
        self.specific_operator = " + "
        self.input_shapes = input_shapes

        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.activation_function = activation_function
        self.constant = constant
        if constant is not None:
            self.constant_size = self.count_elements_array(self.constant)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Broadcast (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Broadcast (size must be int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in Broadcast (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Broadcast (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Broadcast (must be int)"
            msg += "\n"
        for input_shape in self.input_shapes[:, 1:]:
            for shape in input_shape:
                if "int" not in type(shape).__name__:
                    msg += "Error: input_shape in Broadcast (all dim must be int)"
                    msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in Broadcast "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if type(self.constant) is not np.ndarray and self.constant is not None:
            msg += "Error: constant type in Broadcast"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (f"Error: size value in Broadcast "
                    f"({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        if (self.output_channels, self.output_height, self.output_width) != (
                np.max(self.input_shapes[:, 1]), np.max(self.input_shapes[:, 2]), np.max(self.input_shapes[:, 3])):
            msg += (f"Error: non consistency between inputs shape and output shape in Broadcast "
                    f"(({np.max(self.input_shapes[:, 1])},{np.max(self.input_shapes[:, 2])},{np.max(self.input_shapes[:, 3])}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
            msg += "\n"
        for shape in self.input_shapes:
            if (shape[1] != 1 and shape[1] != self.output_channels) or (
                    shape[2] != 1 and shape[2] != self.output_height) or (
                    shape[3] != 1 and shape[3] != self.output_width):
                msg += (f"Error: input shape in Broadcast not broadcastable to shape "
                        f"({self.output_channels}, {self.output_height}, {self.output_width})")
                msg += "\n"
        if msg:
            raise ValueError(msg)

    def forward_path_layer(self: Self, input_arrays: np.ndarray) -> np.ndarray:
        """Compute output of layer."""
        if self.constant is None:
            constant = np.zeros(1)
        else:
            constant = np.reshape(self.constant, self.input_shapes[-1][1:])
            self.input_shapes = np.delete(self.input_shapes, -1, axis=0)
        if len(self.previous_layer) > 1:
            output = np.copy(input_arrays[0]).reshape(self.input_shapes[0][1:])
            for i in range(1, len(input_arrays)):
                output += np.reshape(input_arrays[i], self.input_shapes[i][1:])
        else:
            output = np.reshape(input_arrays, self.input_shapes[0][1:])
        return self.activation_function.compute(output + constant)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
