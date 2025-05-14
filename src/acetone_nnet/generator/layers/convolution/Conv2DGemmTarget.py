"""Convolution direct gemm implementation type definition.

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
from typing_extensions import Self

from acetone_nnet.versioning.layer_factories import conv2d_factory

from . import Conv2D
from .Conv2DGemm import Conv2DGemm


class Conv2DGemmTarget(Conv2DGemm):
    """Implements Conv2D using direct im2col (or im2row) and GeMM."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a Convolution layer with Gemm target specific implementation."""
        super().__init__(**kwargs)

    # None of the tensor ar transposed
    def write_gemm_hw(
        self: Self,
        m: int,
        n: int,
        k: int,
        a: str,
        b: str,
        c: str,
        direct: bool,
    ) -> None:
        """Generate computation code for hw gemm algorithm."""
        mustach_hash = {}

        mustach_hash["direct"] = direct
        mustach_hash["strides"] = self.strides
        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["ldA"] = k
        mustach_hash["B"] = b
        mustach_hash["ldB"] = n
        mustach_hash["C"] = c
        mustach_hash["ldC"] = n
        mustach_hash["activation_function"] = (
            self.activation_function.write_activation_str("output")
        )
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output", self.idx, f"i*{self.ldC} + j"
            )

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(
            self.template_path / "layers" / "Conv" / "template_Conv_gemm_target.c.tpl"
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_im2col(self: Self) -> str:
        """Generate im to col code."""
        output_str = self.previous_layer[0].output_str
        if "output" in output_str:
            output_str = "tensor_temp"

        mustach_hash = {}

        mustach_hash["patches_height"] = self.patches_height
        mustach_hash["kernel_w"] = self.kernel_w
        mustach_hash["kernel_h"] = self.kernel_h
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width
        mustach_hash["strides"] = self.strides
        mustach_hash["pad_top"] = self.pad_top
        mustach_hash["pad_left"] = self.pad_left
        mustach_hash["input_height"] = self.input_height
        mustach_hash["input_width"] = self.input_width
        mustach_hash["road"] = self.path
        mustach_hash["patches_width"] = self.patches_width
        mustach_hash["output_str"] = output_str

        with open(
            self.template_path / "layers" / "Conv" / "template_im2col.c.tpl"
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["size"] = self.size
        mustach_hash["road"] = self.path

        mustach_hash["patch_building_code"] = self.write_im2col()
        mustach_hash["patches_size"] = self.nb_filters * self.patches_width
        mustach_hash["gemm_code"] = self.write_gemm_hw(
            self.nb_filters,
            self.patches_width,
            self.patches_height,
            f"weights_{self.name}_{self.idx:02d}",
            f"output_{self.path}",
            "tensor_temp",
            False,
        )

        if "cst" not in self.previous_layer[0].output_str:
            mustach_hash["cst"] = True
            mustach_hash["input_size"] = (
                self.input_channels * self.input_height * self.input_width
            )

        with open(
            self.template_path / "layers" / "Conv" / "template_Conv_std_gemm.c.tpl"
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def conv2d_gemm_target_implementation(
    old_layer: Conv2D,
    conv_algo: str,
) -> Conv2DGemmTarget:
    """Create a Conv2DGemmTarget layer using the attributes of old_layer."""
    return Conv2DGemmTarget(
        idx=old_layer.idx,
        conv_algorithm=conv_algo,
        size=old_layer.size,
        padding=old_layer.padding,
        strides=old_layer.strides,
        kernel_h=old_layer.kernel_h,
        kernel_w=old_layer.kernel_w,
        dilation_rate=old_layer.dilation_rate,
        nb_filters=old_layer.nb_filters,
        input_shape=[
            1,
            old_layer.input_channels,
            old_layer.input_height,
            old_layer.input_width,
        ],
        output_shape=[
            1,
            old_layer.output_channels,
            old_layer.output_height,
            old_layer.output_width,
        ],
        weights=old_layer.weights,
        biases=old_layer.biases,
        activation_function=old_layer.activation_function,
    )


# conv2d_factory.register_implementation("gemm_target", conv2d_gemm_target_implementation)
