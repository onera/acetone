"""Input Layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import input_factory

from .Input import InputLayer


class InputLayerDefault(InputLayer):
    """Input layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Input Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["road"] = self.path

        if self.data_format == "channels_last" and len(self.input_shape) == 4:
            mustach_hash["channels_last"] = True
            mustach_hash["input_channels"] = self.output_channels
            mustach_hash["input_height"] = self.output_height
            mustach_hash["input_width"] = self.output_width
        else:
            mustach_hash["size"] = self.size

        with open(
                self.template_path / "layers" / "template_Input_Layer.c.tpl",
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def input_default_implementation(
        old_layer: InputLayer,
        version: str
) -> InputLayerDefault:
    """Create a InputLayer_Default layer using the attributes of old_layer."""
    return InputLayerDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shape=old_layer.input_shape,
        data_format=old_layer.data_format,
    )


input_factory.register_implementation(
    None,
    input_default_implementation,
)
input_factory.register_implementation(
    "default",
    input_default_implementation,
)
