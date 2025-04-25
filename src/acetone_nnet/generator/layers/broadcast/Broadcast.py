"""Broadcast layer base type definition.

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
import pystache
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


# The class of the Layers which compute operation with broadcast numpy style
# attribut: none
# input: a list of tensor
# output: the resultant tensor
class Broadcast(Layer):
    """Broadcast layer base class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shapes: list[np.ndarray],
            output_shape: list[int],
            activation_function: ActivationFunctions,
            constant: np.ndarray | float | None = None,
    ) -> None:
        """Instantiate a Broadcast base layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ""
        self.input_shapes = input_shapes

        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.specific_operator = ""
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

    # Go through all the indices and do the operation
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[k]")

        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width

        start = 0
        if self.name == "Maximum":
            start = 1
            mustach_hash["output_str0"] = self.previous_layer[0].output_str
            mustach_hash["input_width0"] = self.input_shapes[0][3]
            mustach_hash["input_height0"] = self.input_shapes[0][2]
            mustach_hash["input_channels0"] = self.input_shapes[0][1]
            mustach_hash["max"] = True
        elif self.name == "Minimum":
            start = 1
            mustach_hash["output_str0"] = self.previous_layer[0].output_str
            mustach_hash["input_width0"] = self.input_shapes[0][3]
            mustach_hash["input_height0"] = self.input_shapes[0][2]
            mustach_hash["input_channels0"] = self.input_shapes[0][1]
            mustach_hash["min"] = True
        elif self.name == "Average":
            mustach_hash["Average"] = True
            mustach_hash["prev_size"] = len(self.previous_layer)
        else:
            mustach_hash["other"] = True

        mustach_hash["broadcast"] = []
        for k in range(start, len(self.previous_layer)):
            previous_dict = {}
            previous_dict["output_str"] = self.previous_layer[k].output_str
            previous_dict["input_width"] = self.input_shapes[k][3]
            previous_dict["input_height"] = self.input_shapes[k][2]
            previous_dict["input_channels"] = self.input_shapes[k][1]
            if k != len(self.previous_layer) - 1:
                previous_dict["operator"] = self.specific_operator
            mustach_hash["broadcast"].append(previous_dict)

        if self.constant is not None:
            constant_dict = {}
            constant_dict["cst_width"] = self.input_shapes[-1][3]
            constant_dict["cst_height"] = self.input_shapes[-1][2]
            constant_dict["cst_channels"] = self.input_shapes[-1][1]
            constant_dict["operator"] = self.specific_operator
            mustach_hash["constant"] = [constant_dict]

        with open(self.template_path / "layers" / "template_Broadcast.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    @abstractmethod
    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
