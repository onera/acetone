"""Conv versioning manager.

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
from collections.abc import Callable

from ...code_generator import (
    Conv2D,
    Conv2D_6loops,
    Conv2D_indirect_gemm,
    Conv2D_std_gemm,
)


def Conv2D_6loops_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2D_6loops:
    """Create a Conv2D_6loops layer using the attributs of old_layer."""
    return Conv2D_6loops(
        idx=old_layer.idx,
        conv_algorithm=conv_algo,
        size=old_layer.size,
        padding=old_layer.padding,
        strides=old_layer.strides,
        kernel_h=old_layer.kernel_h,
        kernel_w=old_layer.kernel_w,
        dilation_rate=old_layer.dilation_rate,
        nb_filters=old_layer.nb_filters,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        weights=old_layer.weights,
        biases=old_layer.biases,
        activation_function=old_layer.activation_function,
    )


def Conv2D_indirect_gemm_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2D_indirect_gemm:
    """Create a Conv2D_indirect_gemm layer using the attributs of old_layer."""
    return Conv2D_indirect_gemm(
        idx=old_layer.idx,
        conv_algorithm=conv_algo,
        size=old_layer.size,
        padding=old_layer.padding,
        strides=old_layer.strides,
        kernel_h=old_layer.kernel_h,
        kernel_w=old_layer.kernel_w,
        dilation_rate=old_layer.dilation_rate,
        nb_filters=old_layer.nb_filters,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        weights=old_layer.weights,
        biases=old_layer.biases,
        activation_function=old_layer.activation_function,
    )


def Conv2D_std_gemm_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2D_std_gemm:
    """Create a Conv2D_std_gemm layer using the attributs of old_layer."""
    return Conv2D_std_gemm(
        idx=old_layer.idx,
        conv_algorithm=conv_algo,
        size=old_layer.size,
        padding=old_layer.padding,
        strides=old_layer.strides,
        kernel_h=old_layer.kernel_h,
        kernel_w=old_layer.kernel_w,
        dilation_rate=old_layer.dilation_rate,
        nb_filters=old_layer.nb_filters,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        weights=old_layer.weights,
        biases=old_layer.biases,
        activation_function=old_layer.activation_function,
    )


Conv2DVariant = Callable[[Conv2D, str], Conv2D]


class Conv2DFactory:
    """Build Conv2D implementation layers."""

    def __init__(self) -> None:
        """Build default convolution layer factory."""
        self.implementations: dict[str | None, Conv2DVariant] = {
            None: Conv2D_6loops_implementation,
            "6loops": Conv2D_6loops_implementation,
            "indirect_gemm": Conv2D_indirect_gemm_implementation,
            "std_gemm": Conv2D_std_gemm_implementation,
        }

    def register_implementation(self, name: str, variant: Conv2DVariant) -> None:
        """Register a new Conv2D variant."""
        if name in self.implementations:
            msg = f"Convolution variant {name} already exists."
            raise KeyError(msg)
        self.implementations[name] = variant

    def __call__(self, layer: Conv2D, version: str) -> Conv2D:
        """Create a Convolution implementation layer for the required implementation."""
        conv_algo = version
        if version != "6loops":
            version = version[:-3]

        return self.implementations[version](layer, conv_algo)


conv2d_factory = Conv2DFactory()
