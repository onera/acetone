"""Activation Layers type definition.

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


class ActivationLayer(Layer):
    """Abstract class for activation layers."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            activation_function: ActivationFunctions,
    ) -> None:
        """Initiate the class."""
        super().__init__()

        self.idx = idx
        self.size = size
        self.activation_function = activation_function
        self.name = activation_function.name.capitalize()

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Activation Layer (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Activation Layer (size must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += "Error: activation_function type in Activation Layer (activation_function must be ActivationFunctions)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.op_type not in [
            "sigmoid",
            "relu",
            "leakyrelu",
            "hyperb_tan",
            "linear",
            "Exponential",
            "Logarithm",
            "Clip",
        ]:
            msg += f"Error: op_type value in Activation Layer ({self.conv_algorithm})"
            msg += "\n"
        if msg:
            raise ValueError(msg)


    def forward_path_layer(
            self: Self,
            inputs: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """Compute output of layer."""
        return self.activation_function.compute(inputs)


    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        input_str = self.previous_layer[0].output_str

        template = "    //{{name}}_{{idx}}\n"
        template += "    for (k = 0; k < {{size}}; ++k) output_{{path}}[k] = {{{function_str}}}\n"

        function_str = self.activation_function.write_activation_str(f"{input_str}[k]")
        mustach_hash = {
            "name": self.name,
            "idx": self.idx,
            "size": self.size,
            "function_str": function_str,
        }
        return pystache.render(template, mustach_hash)
