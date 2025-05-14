"""Gemm Layer type definition.

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


# The layer which compute the general matrix multiplication
# input: weight tesnsor W and bias tensor B, input tensor T. The tensor must be of 2D
# data: alpha and beta constante used in the operation, transpo un tuple saying if the tensor T or W must be transposed before the operation
# output: The result of the operation """alpha*T*W + beta*B"""
class Gemm(Layer):
    """Gemm layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            alpha: float | int,
            beta: float | int,
            transA: bool | int,
            transB: bool | int,
            weights: np.ndarray,
            bias: np.ndarray,
            input_shape: list,
            output_shape: list,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a Gemm layer."""
        super().__init__()
        self.name = "Gemm"
        self.idx = idx
        self.size = size

        if alpha != 1:
            self.alpha = [alpha]
        else:
            self.alpha = []
        if beta != 1:
            self.beta = [beta]
        else:
            self.beta = []

        self.transpo = (transA, transB)

        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        if input_shape:
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
        else:
            self.input_height = 1
            self.input_width = 1

        self.weights = weights
        self.biases = bias
        self.activation_function = activation_function
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Gemm (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Gemm (size must be int)"
            msg += "\n"
        if type(self.alpha[0]) is not float and type(self.alpha[0]) is int:
            msg += "Error: alpha type in Gemm (alpha must be int or float)"
            msg += "\n"
        if type(self.beta[0]) is not float and type(self.beta[0]) is int:
            msg += "Error: beta type in Gemm (beta must be int or float)"
            msg += "\n"
        if any(type(self.transpo[i]) is not int and type(self.transpo[i]) is not bool for i in range(2)):
            msg += "Error: transpose type in Gemm (must be boolean or int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Gemm (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Gemm (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Gemm (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Gemm (must be int)"
            msg += "\n"
        if type(self.weights) is not np.ndarray:
            msg += "Error: weights in Gemm (weights must be an numpy array)"
            msg += "\n"
        if type(self.biases) is not np.ndarray:
            msg += "Error: biases in Gemm (biases must be an numpy array)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += "Error: activation function type in Gemm (activation function must be a sub-classe of acetone_nnet Activation Function)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        if self.size != self.output_height * self.output_width:
            msg += f"Error: size value in Gemm ({self.size}!={self.output_height * self.output_width})"
            msg += "\n"
        shape = self.input_height if self.transpo[0] else self.input_width
        if self.weights.shape[self.transpo[1]] != shape:
            msg += f"Error: non consistency between weight shape and input shape in Gemm ({self.weights.shape[self.transpo[1]]}!={shape})"
            msg += "\n"
        shape = self.input_width if self.transpo[0] else self.input_height
        if self.output_height != shape:
            msg += f"Error: non consistency between input shape and output shape in Gemm ({self.output_height}!={shape})"
            msg += "\n"
        if self.output_width != self.weights.shape[1 - self.transpo[1]]:
            msg += f"Error: non consistency between output shape and output weight in Gemm ({self.output_width}!={self.weights.shape[1 - self.transpo[1]]})"
            msg += "\n"
        if any(self.biases.shape[i] != 1 and self.biases.shape[i] != output_shape[3 - i] for i in
               range(len(self.biases.shape))):
            msg = f"Error: biases in Gemm not broadcastable to dim ({self.output_height},{self.output_width})"
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
        input_array = input_array.reshape(self.input_height, self.input_width)
        if self.transpo[0]:
            input_array = input_array.transpose()

        if self.transpo[1]:
            self.weights = self.weights.transpose()

        return self.activation_function.compute(
            self.alpha * np.dot(input_array, self.weights) + self.beta * self.biases)
