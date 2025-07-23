"""Dense layer type definition.

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
from acetone_nnet.ir import Layer


class Dense(Layer):
    """Dense layer class."""

    def __init__(
        self: Self,
        original_name: str,
        idx: int,
        size: int,
        weights: np.ndarray,
        biases: np.ndarray,
        activation_function: ActivationFunctions,
    ) -> None:
        """Build a Dense layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "Dense"
        if original_name == "":
            self.original_name = f"{self.name}_{self.idx}"
        else:
            self.original_name = original_name
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function

        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.weights) is not np.ndarray:
            msg += "Error: weights in Dense (weights must be an numpy array)"
            msg += "\n"
        if type(self.biases) is not np.ndarray:
            msg += "Error: biases in Dense (biases must be an numpy array)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.weights.shape[-1]:
            msg += (
                f"Error: non consistency between weight shape and output shape in Dense "
                f"({self.size}!={self.weights.shape[-1]})"
            )
            msg += "\n"
        if self.size != self.biases.shape[0]:
            msg += (
                f"Error: non consistency between biases shape and output shape in Dense "
                f"({self.size}!={self.weights.shape[-1]})"
            )
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
        input_array = input_array.reshape(self.previous_layer[0].size)
        return self.activation_function.compute(
            np.dot(input_array, self.weights) + self.biases,
        )
