"""BatchNormalization layer type definition.

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


class BatchNormalization(Layer):
    """BatchNormalization layer class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shape: list,
            epsilon: float,
            scale: np.ndarray,
            biases: np.ndarray,
            mean: np.ndarray,
            var: np.ndarray,
            activation_function: ActivationFunctions,
    ) -> None:
        """Build a BatchNormalization layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "BatchNormalization"
        self.output_channels = input_shape[1]
        self.output_height = input_shape[2]
        self.output_width = input_shape[3]
        self.epsilon = epsilon
        self.scale = scale
        self.mean = mean
        self.var = var
        self.biases = biases
        self.nb_biases = self.count_elements_array(self.biases)

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in BatchNormalization (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in BatchNormalization (size must be int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in BatchNormalization (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in BatchNormalization (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in BatchNormalization (must be int)"
            msg += "\n"
        if type(self.epsilon) is not float and type(self.epsilon) is int:
            msg += "Error: epsilon type in BatchNormalization (epsilon must be int or float)"
            msg += "\n"
        if type(self.scale) is not np.ndarray:
            msg += "Error: scale in BatchNormalization (scale must be an numpy array)"
            msg += "\n"
        if type(self.mean) is not np.ndarray:
            msg += "Error: mean in BatchNormalization (mean must be an numpy array)"
            msg += "\n"
        if type(self.var) is not np.ndarray:
            msg += "Error: var in BatchNormalization (var must be an numpy array)"
            msg += "\n"
        if type(self.biases) is not np.ndarray:
            msg += "Error: biases in BatchNormalization (biases must be an numpy array)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in BatchNormalization "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (f"Error: size value in BatchNormalization "
                    f"({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        if len(self.scale.shape) != 1 or self.scale.shape[0] != self.output_channels:
            msg += (f"Error: non consistency between the scale shape and the output shape in BatchNormalization "
                    f"({self.scale.shape}!={self.output_channels})")
            msg += "\n"
        if len(self.mean.shape) != 1 or self.mean.shape[0] != self.output_channels:
            msg += (f"Error: non consistency between the mean shape and the output shape in BatchNormalization "
                    f"({self.mean.shape}!={self.output_channels})")
            msg += "\n"
        if len(self.var.shape) != 1 or self.var.shape[0] != self.output_channels:
            msg += (f"Error: non consistency between the var shape and the output shape in BatchNormalization"
                    f" ({self.var.shape}!={self.output_channels})")
            msg += "\n"
        if len(self.biases.shape) != 1 or self.biases.shape[0] != self.output_channels:
            msg += (f"Error: non consistency between the biases shape and the output shape in BatchNormalization "
                    f"({self.biases.shape}!={self.output_channels})")
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
        input_array = np.reshape(
            input_array,
            (self.output_channels, self.output_height, self.output_width),
        )
        output = []
        for i in range(self.output_channels):
            output.append(
                (input_array[i] - self.mean[i]) / np.sqrt(self.var[i] + self.epsilon) * self.scale[i] + self.biases[i])
        return np.array(output)
