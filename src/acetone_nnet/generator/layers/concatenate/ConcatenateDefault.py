"""Concatenate layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import concatenate_factory

from .Concatenate import Concatenate


class ConcatenateDefault(Concatenate):
    """Concatenate layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Concatenate layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        borne_sup = 0
        borne_inf = 0

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[k]")

        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width

        if self.axis == 1:
            mustach_hash["channels"] = True
        elif self.axis == 2:
            mustach_hash["heights"] = True
        elif self.axis == 3:  # noqa: PLR2004
            mustach_hash["widths"] = True

        mustach_hash["concat"] = []
        for k in range(len(self.previous_layer)):
            borne_sup += self.input_shapes[k][self.axis]

            layer_to_concat = {}
            layer_to_concat["input_width"] = self.input_shapes[k][3]
            layer_to_concat["input_height"] = self.input_shapes[k][2]
            layer_to_concat["output_str"] = self.previous_layer[k].output_str
            layer_to_concat["borne_sup"] = borne_sup
            layer_to_concat["borne_inf"] = borne_inf
            mustach_hash["concat"].append(layer_to_concat)

            borne_inf += self.input_shapes[k][self.axis]

        with open(self.template_path / "layers" / "template_Concatenate.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def concatenate_default_implementation(
        old_layer: Concatenate,
        version: str,
) -> ConcatenateDefault:
    """Create a Concatenate_Default layer using the attributes of old_layer."""
    return ConcatenateDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        axis=old_layer.axis,
        input_shapes=old_layer.input_shapes,
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        activation_function=old_layer.activation_function,
    )


concatenate_factory.register_implementation(
    None,
    concatenate_default_implementation,
)

concatenate_factory.register_implementation(
    "default",
    concatenate_default_implementation,
)
