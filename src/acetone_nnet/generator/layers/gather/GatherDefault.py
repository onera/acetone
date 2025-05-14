"""Gather layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import gather_factory

from .Gather import Gather


class GatherDefault(Gather):
    """Gather Default layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Gather Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

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

        mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[position]")

        mustach_hash["indices_len"] = len(self.indices.flatten())
        mustach_hash["input_width"] = self.input_width
        mustach_hash["input_height"] = self.input_height

        if self.axis == 1:
            mustach_hash["channels"] = True
            mustach_hash["output_height"] = self.output_height
            mustach_hash["output_width"] = self.output_width
        elif self.axis == 2:
            mustach_hash["heights"] = True
            mustach_hash["output_channels"] = self.output_channels
            mustach_hash["output_width"] = self.output_width
        elif self.axis == 3:
            mustach_hash["widths"] = True
            mustach_hash["output_channels"] = self.output_channels
            mustach_hash["output_height"] = self.output_height

        if self.activation_function.name == "linear":
            mustach_hash["linear"] = True

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "tensor_temp[position]",
                self.idx,
                "position")

        with open(self.template_path / "layers" / "template_Gather.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def gather_default_implementation(
        old_layer: Gather,
        version: str,
) -> GatherDefault:
    """Create a Gather_Default layer using the parameters of old_layer."""
    return GatherDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        axis=old_layer.axis,
        indices=old_layer.indices,
        input_shape=[1, old_layer.output_channels, old_layer.input_height, old_layer.input_width],
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        activation_function=old_layer.activation_function,
    )


gather_factory.register_implementation(
    None,
    gather_default_implementation,
)
gather_factory.register_implementation(
    "default",
    gather_default_implementation,
)
