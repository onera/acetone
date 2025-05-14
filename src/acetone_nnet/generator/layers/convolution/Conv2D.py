"""Convolution layer dummy layer type definition.

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

import math

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.versioning.layer_factories import conv2d_factory


class Conv2D(Layer):
    """Convolution layer class."""

    def __init__(
        self: Self,
        idx: int,
        conv_algorithm: str,
        size: int,
        padding: str | np.ndarray,
        strides: int,
        kernel_h: int,
        kernel_w: int,
        dilation_rate: int,
        nb_filters: int,
        input_shape: list[int],
        output_shape: list[int],
        weights: np.ndarray,
        biases: np.ndarray,
        activation_function: ActivationFunctions,
    ) -> None:
        """Build a Conv2D layer."""
        super().__init__()
        self.conv_algorithm = conv_algorithm
        self.idx = idx
        self.size = size
        self.name = "Conv2D"
        self.padding = padding
        self.strides = strides
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]

        self.input_shape = [self.input_channels, self.input_height, self.input_width]
        self.output_channels = self.nb_filters

        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function
        self.local_var = "sum"

        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)
        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = (
            self.compute_padding(
                self.padding,
                self.input_height,
                self.input_width,
                self.kernel_h,
                self.kernel_w,
                self.strides,
                self.dilation_rate,
            )
        )

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Conv2D (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Conv2D (size must be int)"
            msg += "\n"
        if type(conv_algorithm) is not str:
            msg += "Error: conv algorithm type in Conv2D (must be str)"
            msg += "\n"
        if type(self.padding) is not str and any(
            type(pad) is not int for pad in self.padding
        ):
            msg += "Error: padding type in Conv2D (must be str or ints)"
            msg += "\n"
        if type(self.strides) is not int:
            msg += "Error: strides type in Conv2D (must be int)"
            msg += "\n"
        if type(self.kernel_h) is not int:
            msg += "Error: kernel_h type in Conv2D (must be int)"
            msg += "\n"
        if type(self.kernel_w) is not int:
            msg += "Error: kernel_w type in Conv2D (must be int)"
            msg += "\n"
        if type(self.dilation_rate) is not int:
            msg += "Error: dilation_rate type in Conv2D (must be int)"
            msg += "\n"
        if type(self.nb_filters) is not int:
            msg += "Error: nb_filters type in Conv2D (must be int)"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Conv2D (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Conv2D (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Conv2D (must be int)"
            msg += "\n"
        if type(self.output_channels) is not int:
            msg += "Error: output channels type in Conv2D (must be int)"
            msg += "\n"
        if type(self.output_height) is not int:
            msg += "Error: output height type in Conv2D (must be int)"
            msg += "\n"
        if type(self.output_width) is not int:
            msg += "Error: output width type in Conv2D (must be int)"
            msg += "\n"
        if type(self.weights) is not np.ndarray:
            msg += "Error: weights in Conv2D (weights must be an numpy array)"
            msg += "\n"
        if type(self.biases) is not np.ndarray:
            msg += "Error: biases in Conv2D (biases must be an numpy array)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += (
                "Error: activation function type in Conv2D "
                "(activation function must be a sub-classe of acetone_nnet Activation Function)"
            )
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (
                f"Error: size value in Conv2D "
                f"({self.size}!={self.output_channels * self.output_height * self.output_width})"
            )
            msg += "\n"
        if self.weights.shape != (
            self.input_channels,
            self.kernel_h,
            self.kernel_w,
            self.nb_filters,
        ):
            msg += (
                f"Error: non consistency between weight shape and operation parameters in Conv2D "
                f"({self.weights.shape}!=({self.input_channels}, {self.kernel_h}, {self.kernel_w}, {self.nb_filters}))"
            )
            msg += "\n"
        if len(self.biases.shape) != 1 or self.biases.shape[0] != self.nb_filters:
            msg += (
                f"Error: non consistency between the var shape and the output shape in Conv2D "
                f"({self.biases.shape}!={self.nb_filters})"
            )
            msg += "\n"
        if (
            self.output_height
            != math.floor(
                (
                    self.input_height
                    + self.pad_bottom
                    + self.pad_top
                    - self.kernel_h
                    - (self.kernel_h - 1) * (self.dilation_rate - 1)
                )
                / self.strides
            )
            + 1
        ):
            msg += (
                f"Error: non consistency between the output height and the parameter of the operation in Conv2D "
                f"({self.output_height}!={math.floor((self.input_height + self.pad_bottom + self.pad_top - self.kernel_h - (self.kernel_h - 1) * (self.dilation_rate - 1)) / self.strides) + 1})"
            )
            msg += "\n"
        if (
            self.output_width
            != math.floor(
                (
                    self.input_width
                    + self.pad_left
                    + self.pad_right
                    - self.kernel_w
                    - (self.kernel_w - 1) * (self.dilation_rate - 1)
                )
                / self.strides
            )
            + 1
        ):
            msg += (
                f"Error: non consistency between the output width and the parameter of the operation in Conv2D "
                f"({self.output_width}!={math.floor((self.input_width + self.pad_left + self.pad_right - self.kernel_w - (self.kernel_w - 1) * (self.dilation_rate - 1)) / self.strides) + 1})"
            )
            msg += "\n"
        if self.conv_algorithm not in conv2d_factory.list_implementations and self.conv_algorithm !="specs":
            msg += f"Error: conv algorithm value in Conv2D ({self.conv_algorithm})"
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
        # Conv for chw
        input_array = input_array.reshape(
            self.input_channels, self.input_height, self.input_width
        )

        output = np.zeros((self.nb_filters, self.output_height, self.output_width))
        print(self.weights.shape)

        if self.pad_right or self.pad_left or self.pad_top or self.pad_bottom:
            input_padded = np.zeros(
                (
                    self.input_channels,
                    self.input_height + self.pad_top + self.pad_bottom,
                    self.input_width + self.pad_left + self.pad_right,
                )
            )

            border_right = None if self.pad_right == 0 else -self.pad_right
            border_bottom = None if self.pad_bottom == 0 else -self.pad_bottom
            input_padded[
                :, self.pad_top : border_bottom, self.pad_left : border_right
            ] = input_array
        else:
            input_padded = input_array

        for f in range(self.nb_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    w = (
                        self.weights[:, :, :, f]
                        * input_padded[
                            :,
                            i * self.strides : i * self.strides + self.kernel_h,
                            j * self.strides : j * self.strides + self.kernel_w,
                        ]
                    )
                    output[f, i, j] = np.sum(w) + self.biases[f]
        return self.activation_function.compute(output)
