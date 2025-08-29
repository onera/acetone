"""Activation Layers type definition.

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

from acetone_nnet.versioning.layer_factories import activation_layer_factory

from .ActivationLayer import ActivationLayer


class ActivationLayerDefault(ActivationLayer):
    """Abstract class for activation layers."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a ActivationLayer layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version


    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        input_str = self.previous_layer[0].output_str

        template  = "    //{{name}}_{{idx}}\n"
        template += "    for (k = 0; k < {{size}}; ++k) output_{{path}}[k] = {{{function_str}}};\n"

        function_str = self.activation_function.write_activation_str(f"{input_str}[k]")
        mustach_hash = {
            "name": self.name,
            "original_name": self.original_name,
            "idx": self.idx,
            "size": self.size,
            "path": self.path,
            "function_str": function_str,
        }
        return pystache.render(template, mustach_hash)

def activation_layer_default_implementation(
        old_layer: ActivationLayer,
        version: str,
) -> ActivationLayerDefault:
    """Create an ActivationLayer_Default layer using the attributes of old_layer."""
    return ActivationLayerDefault(
        version=version,
        original_name=old_layer.original_name,
        idx=old_layer.idx,
        size=old_layer.size,
        activation_function=old_layer.activation_function,
    )

activation_layer_factory.register_implementation(
    None,
    activation_layer_default_implementation,
)
activation_layer_factory.register_implementation(
    "default",
    activation_layer_default_implementation,
)