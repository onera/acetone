"""ReduceSum layer type definition.

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


class ReduceSum(Layer):
    """ReduceSum layer class."""

    def __init__(
        self: Self,
        idx: int,
        size: int,
        axis: tuple[int],
        keepdims: int,
        noop_with_empty_axes: int,
        input_shape: list[int],
        activation_function: ActivationFunctions,
    ) -> None:
        """Instantiate a ReduceSum layer."""
        Layer.__init__(self)
        self.idx = idx
        self.size = size
        self.name = "ReduceSum"

        self.reduce_func = "Sum"

        self.axes = axis
        self.keepdims = bool(keepdims)
        self.noop_with_empty_axes = noop_with_empty_axes

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        if 1 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_channels = 1
        else:
            self.output_channels = self.input_channels

        if 2 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_height = 1
        else:
            self.output_height = self.input_height

        if 3 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_width = 1
        else:
            self.output_width = self.input_width

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Reduce (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Reduce (size must be int)"
            msg += "\n"
        if type(self.keepdims) is not bool:
            msg += "Error: keepdims type in Reduce (size must be bool)"
            msg += "\n"
        if type(self.noop_with_empty_axes) is not int:
            msg += "Error: noop with empty axes type in Reduce (size must be int)"
            msg += "\n"
        if (
            any("int" not in type(axe).__name__ for axe in self.axes)
            or type(self.axes) is not tuple
        ):
            msg += "Error: axes type in Reduce (must be tuple[int])"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Reduce (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Reduce (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Reduce (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += (
                "Error: activation function type in Reduce "
                "(activation function must be a sub-classe of acetone_nnet Activation Function)"
            )
            msg += "\n"
        if msg:
            raise TypeError(msg)


    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(1, self.input_channels, self.input_height, self.input_width)
        if self.axes == ():
            if self.noop_with_empty_axes:
                return input_array
            return np.add.reduce(input_array, axis=None, keepdims=self.keepdims)
        return np.add.reduce(input_array, axis=self.axes, keepdims=self.keepdims)
