"""Pad layer base type definition.

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

from abc import abstractmethod

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


# The Pad Layers
# Pad alongside each dimmensions
# attribut: the mode of padding required
# input: a tensor to be padded, the desired pads, the value of teh constant if mode == constant
# output:the resized tensor
######################### cf https://onnx.ai/onnx/operators/onnx__Pad.html for the doc
class Pad(Layer):
    """Pad layer base class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            pads: np.ndarray,
            constant_value: float,
            axes: np.ndarray | list[int],
            input_shape: list[int],
            activation_function: ActivationFunctions,
    ) -> None:
        """Instantiate a Pas base layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.pads = pads
        self.constant_value = constant_value
        self.axes = axes
        self.name = "Pad"
        self.input_shape = input_shape
        self.output_channels = input_shape[1] + pads[1] + pads[5]
        self.output_height = input_shape[2] + pads[2] + pads[6]
        self.output_width = input_shape[3] + pads[3] + pads[7]
        self.mode = ""
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Pad (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Pad (size must be int)"
            msg += "\n"
        if any(type(pad) is not int for pad in self.pads):
            msg += "Error: pads type in Pad (must be int)"
            msg += "\n"
        if type(self.constant_value) is not float and type(self.constant_value) is not int:
            msg += "Error: constant value type in Pad (must be float or int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in Pad (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Pad (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Pad (must be int)"
            msg += "\n"
        if any(type(shape) is not int for shape in self.input_shape[1:]):
            msg += "Error: input shape type in Pad (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in Pad "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg = (
                f"Error: size value in Pad "
                f"({self.size}!={self.output_channels * self.output_height * self.output_width})"
            )
            raise ValueError(msg)
        for axe in self.axes:
            if axe < 0 or axe >= 4:
                msg = (
                    f"Error: axe out of bound in Pad "
                    f"({axe}for tensor in 4 dimension with first dimension unused)"
                )
                raise ValueError(msg)

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(self.input_shape[1], self.input_shape[2], self.input_shape[3])
        nb_dim = len(self.pads) // 2
        pad_width = [(self.pads[i], self.pads[i + nb_dim]) for i in
                     range(1, nb_dim)]  # Constructing the pads accordingly to the numpy nomenclature
        return self.activation_function.compute(
            np.pad(input_array, pad_width=pad_width, mode=self.mode, constant_values=self.constant_value),
        )

    @abstractmethod
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
