"""Convolution indirect gemm implementation type definition.

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

from acetone_nnet.versioning.layer_factories import conv2d_factory

from . import Conv2D
from .Conv2DGemm import Conv2DGemm


class Conv2DIndirectGemm(Conv2DGemm):
    """Implements Conv2D using indirect im2col (or im2row) and GeMM."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a Convolution layer with indirect gemm implementation."""
        super().__init__(**kwargs)

    def create_ppatches(self: Self) -> str:
        """Generate the new matrix to multiply in the Gemm algorithm."""
        if self.pad_right or self.pad_left or self.pad_bottom or self.pad_top:
            self.input_h_padded = self.input_height + self.pad_top + self.pad_bottom
            self.input_w_padded = self.input_width + self.pad_left + self.pad_right

            start_idx = np.arange(self.kernel_h)[:, None] * self.input_w_padded + np.arange(self.kernel_w)
            c = self.input_h_padded * self.input_w_padded * np.arange(self.input_channels)
            start_idx = (c[:, None] + start_idx.ravel()).reshape((-1, self.kernel_h, self.kernel_w))
            offset_idx = np.arange(self.output_height * self.strides, step=self.strides)[:,
                         None] * self.input_w_padded + np.arange(self.output_width * self.strides, step=self.strides)
            idx_padded_input = (start_idx.ravel()[:, None] + offset_idx.ravel()).flatten()

            idx_of_zeros = []
            j_zeros = np.concatenate(
                (np.arange(self.pad_left), np.arange(self.pad_right) + (self.input_w_padded - self.pad_right)))
            i_zeros = np.concatenate(
                (np.arange(self.pad_top), np.arange(self.pad_bottom) + (self.input_h_padded - self.pad_bottom)))
            for c in range(self.input_channels):
                for i in range(self.input_h_padded):
                    for j in range(self.input_w_padded):
                        if np.isin(i, i_zeros) or np.isin(j, j_zeros):
                            idx_of_zeros.append(j + self.input_w_padded * (i + self.input_h_padded * c))

            idx_padded_input = np.where(np.isin(idx_padded_input, idx_of_zeros), np.nan, idx_padded_input)
            _, idx_padded_input = np.unique(idx_padded_input, return_inverse=True)
            self.ppatches = np.where(idx_padded_input == self.input_channels * self.input_height * self.input_width,
                                     np.nan, idx_padded_input)

        else:
            start_idx = np.arange(self.kernel_h)[:, None] * self.input_width + np.arange(self.kernel_w)
            c = self.input_height * self.input_width * np.arange(self.input_channels)
            start_idx = (c[:, None] + start_idx.ravel()).reshape((-1, self.kernel_h, self.kernel_w))
            offset_idx = np.arange(self.output_height * self.strides, step=self.strides)[:,
                         None] * self.input_width + np.arange(self.output_width * self.strides, step=self.strides)
            self.ppatches = (start_idx.ravel()[:, None] + offset_idx.ravel()).flatten()

        if "gemm_nt" in self.conv_algorithm or "gemm_tt" in self.conv_algorithm:
            self.ppatches = self.ppatches.reshape((self.patches_height, self.patches_width)).transpose().flatten()

        output_str = self.previous_layer[0].output_str

        s = "\n        {"
        for i in range(len(self.ppatches)):
            if np.isnan(self.ppatches[i]):
                s += "&zero, "
            else:
                s += "&" + output_str + "[" + str(int(self.ppatches[i])) + "], "

        print(self.name, self.idx, "patches size:", self.ppatches.shape)

        s = s[:-2]
        s += "}"

        return s

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(self.local_var)

        gemm_code = self.algo_gemm_mapping[self.conv_algorithm](
            self.nb_filters,
            self.patches_width,
            self.patches_height,
            f"weights_{self.name}_{self.idx:02d}",
            f"ppatches_{self.name}_{self.idx:02d}",
            f"output_{self.path}",
            True)
        mustach_hash["gemm_code"] = gemm_code

        if "cst" not in self.previous_layer[0].output_str:
            mustach_hash["cst"] = True
        mustach_hash["prev_size"] = self.input_channels * self.input_height * self.input_width

        with open(self.template_path / "layers" / "Conv" / "template_Conv_indirect_gemm.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def conv2d_indirect_gemm_implementation(
        old_layer: Conv2D,
        conv_algo: str,
) -> Conv2DIndirectGemm:
    """Create a Conv2D_indirect_gemm layer using the attributes of old_layer."""
    return Conv2DIndirectGemm(
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


conv2d_factory.register_implementation("indirect_gemm_nn", conv2d_indirect_gemm_implementation)
conv2d_factory.register_implementation("indirect_gemm_tn", conv2d_indirect_gemm_implementation)
conv2d_factory.register_implementation("indirect_gemm_nt", conv2d_indirect_gemm_implementation)
conv2d_factory.register_implementation("indirect_gemm_tt", conv2d_indirect_gemm_implementation)
