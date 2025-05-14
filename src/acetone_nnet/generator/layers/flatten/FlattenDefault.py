"""Flatten layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import flatten_factory

from .Flatten import Flatten


class FlattenDefault(Flatten):
    """Flatten layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Flatten layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        if self.data_format == "channels_last":
            mustach_hash["channels_last"] = True
            mustach_hash["input_channels"] = self.input_shape[1]
            mustach_hash["input_height"] = self.input_shape[2]
            mustach_hash["input_width"] = self.input_shape[3]
            mustach_hash["name"] = self.name
            mustach_hash["idx"] = f"{self.idx:02d}"
            mustach_hash["path"] = self.path
            mustach_hash["size"] = self.size

        with open(self.template_path / "layers" / "template_Flatten.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def flatten_default_implementation(
        old_layer: Flatten,
        version: str,
) -> FlattenDefault:
    """Create a Flatten_Default layer using the attributes of old_layer."""
    return FlattenDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shape=old_layer.input_shape,
        data_format=old_layer.data_format,
    )


flatten_factory.register_implementation(
    None,
    flatten_default_implementation,
)

flatten_factory.register_implementation(
    "default",
    flatten_default_implementation,
)
