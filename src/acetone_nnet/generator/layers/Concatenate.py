"""Concatenate layer type definition.

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


# Concatenate two tensor alongside a given axis
# attribut: axis alongside of which the concatenation will be done
# input: a list of tensor to concatenate
# output: the concatenated tensor
class Concatenate(Layer):
    """Concatenate layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            axis: int,
            input_shapes: list,
            output_shape: list,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a Concatenate layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = "Concatenate"
        self.axis = axis
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Concatenate (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Concatenate (size must be int)"
            msg += "\n"
        for input_shape in self.input_shapes:
            if any(type(shape) is not int for shape in input_shape[1:]):
                msg += "Error: input_shape in Concatenate (all dim must be int)"
                msg += "\n"
        if type(self.axis) is not int:
            msg += "Error: axis type in Concatenate (axis must be int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in Concatenate (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Concatenate (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Concatenate (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += "Error: activation function type in Concatenate (activation function must be a sub-classe of acetone_nnet Activation Function)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += ("Error: size value in Concatenate "
                    "({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        if self.axis not in [1, 2, 3]:
            msg += ("Error: axis out of bound in Concatenate "
                    "({axis} for tensor in 4 dimension with first dimension unused)")
            msg += "\n"
        for i in range(len(self.input_shapes)):
            if self.axis != 1 and self.output_channels != self.input_shapes[i][1]:
                msg += ("Error: non consistency between the tensors shapes in Concatenate "
                        "({self.input_shapes[i][1:]}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
                msg += "\n"
            if self.axis != 2 and self.output_height != self.input_shapes[i][2]:
                msg += ("Error: non consistency between the tensors shapes in Concatenate "
                        "({self.input_shapes[i][1:]}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
                msg += "\n"
            if self.axis != 3 and self.output_width != self.input_shapes[i][3]:
                msg += ("Error: non consistency between the tensors shapes in Concatenate "
                        "({self.input_shapes[i][1:]}!=({self.output_channels}, {self.output_height}, {self.output_width}))")
                msg += "\n"

        if self.axis == 1 and self.output_channels != sum(
                [self.input_shapes[i][1] for i in range(len(self.input_shapes))]):
            msg += ("Error: non consistency between the tensors shapes and the output shape in Concatenate "
                    "({sum(self.input_shapes[:][1])}!={self.output_channels})")
            msg += "\n"
        if self.axis == 2 and self.output_height != sum(
                [self.input_shapes[i][2] for i in range(len(self.input_shapes))]):
            msg += ("Error: non consistency between the tensors shapes and the output shape in Concatenate "
                    "({sum(self.input_shapes[:][2])}!={self.output_height})")
            msg += "\n"
        if self.axis == 3 and self.output_width != sum(
                [self.input_shapes[i][3] for i in range(len(self.input_shapes))]):
            msg += ("Error: non consistency between the tensors shapes and the output shape in Concatenate "
                    "({sum(self.input_shapes[:][3])}!={self.output_width})")
            msg += "\n"
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        borne_sup = 0
        borne_inf = 0

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

        if self.axis == 1:
            mustach_hash["channels"] = True
        elif self.axis == 2:
            mustach_hash["heights"] = True
        elif self.axis == 3:  # noqa: PLR2004
            mustach_hash["widths"] = True

        mustach_hash["concat"] = []
        for k in range(len(self.previous_layer)):
            borne_sup += self.input_shapes[k][self.axis]

            layer_to_concat = {}
            layer_to_concat["input_width"] = self.input_shapes[k][3]
            layer_to_concat["input_height"] = self.input_shapes[k][2]
            layer_to_concat["output_str"] = self.previous_layer[k].output_str
            layer_to_concat["borne_sup"] = borne_sup
            layer_to_concat["borne_inf"] = borne_inf
            mustach_hash["concat"].append(layer_to_concat)

            borne_inf += self.input_shapes[k][self.axis]

        with open(self.template_path + "layers/template_Concatenate.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def forward_path_layer(
            self: Self,
            inputs_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        output = inputs_array[0]
        output = output.reshape(
            self.input_shapes[0][1],
            self.input_shapes[0][2],
            self.input_shapes[0][3],
        )
        for i in range(1, len(inputs_array)):
            input_array = inputs_array[i]
            input_array = input_array.reshape(
                self.input_shapes[i][1],
                self.input_shapes[i][2],
                self.input_shapes[i][3],
            )
            output = np.concatenate((output, input_array), axis=self.axis - 1)
        return self.activation_function.compute(output)