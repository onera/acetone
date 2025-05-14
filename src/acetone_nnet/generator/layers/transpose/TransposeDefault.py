"""Transpose Layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import transpose_factory

from .Transpose import Transpose


class TransposeDefault(Transpose):
    """Transpose layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Tile Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["size"] = self.size
        mustach_hash["road"] = self.path
        mustach_hash["output_str"] = output_str

        mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[k]")

        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width
        mustach_hash["input_height"] = self.input_height
        mustach_hash["input_width"] = self.input_width

        indices = ["Batch", "f", "i", "j"]
        if self.perm[1:] == [2, 3, 1]:
            mustach_hash["a"] = indices[self.perm[1]]
            mustach_hash["b"] = indices[self.perm[3]]
            mustach_hash["c"] = indices[self.perm[2]]
        elif self.perm[1:] == [3, 1, 2]:
            mustach_hash["a"] = indices[self.perm[2]]
            mustach_hash["b"] = indices[self.perm[1]]
            mustach_hash["c"] = indices[self.perm[3]]
        else:
            mustach_hash["a"] = indices[self.perm[3]]
            mustach_hash["b"] = indices[self.perm[2]]
            mustach_hash["c"] = indices[self.perm[1]]

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output_" + str(self.path) + "[j]",
                self.idx,
                "j")

        with open(self.template_path / "layers" / "template_Transpose.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def transpose_default_implementation(
        old_layer: Transpose,
        version: str,
) -> TransposeDefault:
    """Create a Transpose_Default layer using the attributes of old_layer."""
    return TransposeDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        perm=old_layer.perm,
        activation_function=old_layer.activation_function,
    )


transpose_factory.register_implementation(
    None,
    transpose_default_implementation,
)
transpose_factory.register_implementation(
    "default",
    transpose_default_implementation,
)
