"""Convolution 6 loops direct with memory blocking implementation type definition.

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
import logging

def transform_to_6d_blocked_layout(weight, block_k, block_c):
    # KCHW -> [NB_K][NB_C][KH][KW][BLOCK_C][BLOCK_K]
    K, C, KH, KW = weight.shape
    nb_k, nb_c = K // block_k, C // block_c
    layout = weight.copy()
    layout = layout.reshape(nb_k, block_k, nb_c, block_c, KH, KW)
    return layout.transpose(0, 2, 4, 5, 3, 1).copy("C")

class Conv2Ddirect_block(Conv2D):
    """Implements Conv2D using the six-loops algorithm (direct conv) with blocking of channels."""

    def __init__(self: Self, **kwargs: Any) -> None:
        """Build a Convolution layer with 6 loops implementation."""
        super().__init__(**kwargs)
        self.layout = transform_to_6d_blocked_layout(self.weights,min(64,self.nb_filters),self.input_channels)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["original_name"] = self.original_name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["output_str"] = output_str
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size
        if self.activation_function.name != "linear":
            mustach_hash["activation_function"] = self.activation_function.write_activation_str("o_ptr[k]")

        assert(self.pad_left == self.pad_top)
        assert(self.dilation_rate == 1)
        params={
            "H_in":self.input_height,
            "W_in":self.input_width,
            "C":self.input_channels,
            "H_out":self.output_height,
            "W_out":self.output_width,
            "K":self.nb_filters,
            "KH":self.kernel_h,
            "KW":self.kernel_w,
            "STRIDE":self.strides,
            "PAD":self.pad_left,
            "BLOCK_C":self.input_channels,
            "BLOCK_K":min(64,self.nb_filters),
        }
        params["NB_BLOCK_C"] = self.input_channels // params["BLOCK_C"]
        params["NB_BLOCK_K"] = self.nb_filters // params["BLOCK_K"]

        mustach_hash.update(params)

        with open(self.template_path / "layers" / "Conv" / "template_Conv_direct_block.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def conv2d_dirblock_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2Ddirect_block:
    """Create a Conv2D_6loops layer using the attributes of old_layer."""
    return Conv2Ddirect_block(
        idx=old_layer.idx,
        original_name=old_layer.original_name,
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

conv2d_factory.register_implementation("direct_block", conv2d_dirblock_implementation)
