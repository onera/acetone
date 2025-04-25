"""AddBias layer type definition.

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


class AddBias(Layer):
    """AddBias layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            biases: np.ndarray,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a AddBias layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "AddBias"
        self.biases = biases
        self.nb_biases = self.count_elements_array(self.biases)
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in AddBias (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in AddBias (size must be int)"
            msg += "\n"
        if type(self.biases) is not np.ndarray:
            msg += "Error: biases in AddBias (biases must be an numpy array)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in AddBias "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if len(self.biases.shape) != 1 or self.biases.shape[0] != self.size:
            msg += (f"Error:non consistency between the biases shape and the output shape in AddBias"
                    f" ({self.biases.shape[0]}!={self.size})")
            msg += "\n"
        if msg:
            raise ValueError(msg)

    # Go through all the indices and do the operation
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["output_str"] = output_str
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(
            f"output_{self.path}[i]")

        if self.activation_function.name == "linear":
            mustach_hash["linear"] = True

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                f"output_{self.path}[i]",
                self.idx,
                "i")

        with open(self.template_path / "layers" / "template_AddBias.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(self.previous_layer[0].size)

        return self.activation_function.compute(input_array + self.biases)
