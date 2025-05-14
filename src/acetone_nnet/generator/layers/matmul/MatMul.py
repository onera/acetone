"""MatMul Layer type definition.

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


class MatMul(Layer):
    """MatMul layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shapes: list,
            weights: np.ndarray,
            side: int,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a MatMul layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "MatMul"
        self.activation_function = activation_function
        self.local_var = "dotproduct"
        self.side = side
        self.input_shapes = input_shapes

        if weights is not None:
            self.weights = weights
            self.nb_weights = self.count_elements_array(self.weights)

        if self.side == 0:
            self.output_channels = self.input_shapes[1]
            self.output_height = self.input_shapes[-2]
            self.output_width = self.weights.shape[-1]
            self.shared_dimension = self.input_shapes[-1]
        elif self.side == 1:
            self.output_channels = self.input_shapes[1]
            self.output_width = self.input_shapes[-1]
            self.output_height = self.weights.shape[-2]
            self.shared_dimension = self.input_shapes[-2]
        elif self.side == 2:
            self.output_channels = self.input_shapes[0][1]
            self.output_height = self.input_shapes[0][-2]
            self.output_width = self.input_shapes[1][-1]
            self.shared_dimension = self.input_shapes[0][-1]

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg = "Error: idx type in MatMul (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg = "Error: size type in MatMul (size must be int)"
            msg += "\n"
        if type(self.input_shapes[0]) is list:
            for input_shape in self.input_shapes:
                if any(type(shape) is not int for shape in input_shape):
                    msg = "Error: input_shape in MatMul (all dim must be int)"
                    msg += "\n"
        elif any(type(shape) is not int for shape in self.input_shapes):
            msg = "Error: input_shape in MatMul (all dim must be int)"
            msg += "\n"
        if hasattr(self, "weights") and type(self.weights) is not np.ndarray:
            msg = "Error: weights in MatMul (weights must be an numpy array)"
            msg += "\n"
        if type(side) is not int:
            msg = "Error: side type in MatMul (side must be a boolean)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg = "Error: activation function type in MatMul (activation function must be a sub-classe of acetone_nnet Activation Function)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.side == 1:
            if self.weights.shape[-1] != self.input_shapes[-2]:
                msg = f"Error: non consistency between weight shape and input shape in MatMul ({self.weights.shape[-1]}!={self.input_shapes[-2]})"
                msg += "\n"
            if self.size != self.weights.shape[-2] * self.input_shapes[-1] * self.output_channels:
                msg = f"Error: size value in MatMul ({self.size} !={self.weights.shape[-2] * self.input_shapes[-1]})"
                msg += "\n"
        elif self.side == 0:
            if self.weights.shape[-2] != self.input_shapes[-1]:
                msg = f"Error: non consistency between weight shape and input shape in MatMul ({self.weights.shape[-2]}!={self.input_shapes[-1]})"
                msg += "\n"
            if self.size != self.weights.shape[-1] * self.input_shapes[-2] * self.output_channels:
                msg = f"Error: size value in MatMul ({self.size} !={self.weights.shape[-1] * self.input_shapes[-2]})"
                msg += "\n"
        elif self.side == 2:
            if self.input_shapes[1][-2] != self.input_shapes[0][-1]:
                msg = f"Error: non consistency between weight shape and input shape in MatMul ({self.input_shapes[1][-2]}!={self.input_shapes[0][-1]})"
                msg += "\n"
            if self.size != self.input_shapes[1][-1] * self.input_shapes[0][-2] * self.output_channels:
                msg = f"Error: size value in MatMul ({self.size} !={self.input_shapes[1][-1] * self.input_shapes[0][-2]})"
                msg += "\n"
        else:
            msg += f"Error: side value in Matmul (0 is Input*Weight, 1 is Weight*Input, 2 is Input_1*Input_2, {self.side} is not implemented)"
            msg += "\n"
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """Compute output of layer."""
        if self.side == 1:
            input_1 = input_array.reshape(self.input_shapes)
            weights = np.moveaxis(self.weights, 3, 0)
            weights = np.reshape(weights, (1, 1, weights.shape[-1], weights.shape[0]))
            return self.activation_function.compute(np.matmul(weights, input_1))
        if self.side == 0:
            input_1 = input_array.reshape(self.input_shapes)
            weights = np.moveaxis(self.weights, 3, 0)
            weights = np.reshape(weights, (1, 1, weights.shape[-1], weights.shape[0]))
            return self.activation_function.compute(np.matmul(input_1, weights))
        if self.side == 2:
            input_1 = input_array[0].reshape(self.input_shapes[0])
            input_2 = input_array[1].reshape(self.input_shapes[1])
            return self.activation_function.compute(np.matmul(input_1, input_2))
        return np.array([])  # Case should not be happening
