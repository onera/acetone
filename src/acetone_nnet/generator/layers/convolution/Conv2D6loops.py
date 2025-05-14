"""Convolution 6 loops implementation type definition.

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

import pystache
from typing_extensions import Any, Self

from acetone_nnet.versioning.layer_factories import conv2d_factory

from .Conv2D import Conv2D


class Conv2D6loops(Conv2D):
    """Implements Conv2D using the six-loops algorithm (direct conv)."""

    def __init__(self: Self, **kwargs: Any) -> None:
        """Build a Convolution layer with 6 loops implementation."""
        super().__init__(**kwargs)

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

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(self.local_var)

        mustach_hash["nb_filters"] = self.nb_filters
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width
        mustach_hash["input_channels"] = self.input_channels
        mustach_hash["kernel_h"] = self.kernel_h
        mustach_hash["kernel_w"] = self.kernel_w
        mustach_hash["strides"] = self.strides
        mustach_hash["dilation_rate"] = self.dilation_rate
        mustach_hash["pad_left"] = self.pad_left
        mustach_hash["pad_top"] = self.pad_top
        mustach_hash["input_height"] = self.input_height
        mustach_hash["input_width"] = self.input_width

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                self.local_var,
                self.idx,
                f"j + {self.output_width}*(i + {self.output_height}*f)")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Conv" / "template_Conv_6loops.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def conv2d_6loops_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2D6loops:
    """Create a Conv2D_6loops layer using the attributes of old_layer."""
    return Conv2D6loops(
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


conv2d_factory.register_implementation(None, conv2d_6loops_implementation)
conv2d_factory.register_implementation("6loops", conv2d_6loops_implementation)
