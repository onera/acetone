"""Dense layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import dense_factory

from .Dense import Dense


class DenseDefault(Dense):
    """Dense layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Dense layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        # Variable indicating under which name the input tensor is
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["output_str"] = output_str
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = (
            self.activation_function.write_activation_str(self.local_var)
        )

        mustach_hash["prev_size"] = self.previous_layer[0].size

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                self.local_var,
                self.idx,
                "i",
            )

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "template_Dense.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def dense_default_implementation(
        old_layer: Dense,
        version: str,
) -> DenseDefault:
    """Create a Dense_Default layer using the attributes of old_layer."""
    return DenseDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        weights=old_layer.weights,
        biases=old_layer.biases,
        activation_function=old_layer.activation_function,
    )


dense_factory.register_implementation(
    None,
    dense_default_implementation,
)

dense_factory.register_implementation(
    "default",
    dense_default_implementation,
)
