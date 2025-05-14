"""MatMul Layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import matmul_factory

from .MatMul import MatMul


class MatMulDefault(MatMul):
    """MatMul layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a MatMul Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        if self.activation_function.name != "linear":
            mustach_hash["non_linear"] = True
            mustach_hash["activation_function"] = self.activation_function.write_activation_str(
                f"tensor_temp[j + {self.output_width}*(i + {self.output_height}*f)]")

        mustach_hash["shared_dimension"] = self.shared_dimension
        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width

        if self.side == 0:
            mustach_hash["output_str_left"] = self.previous_layer[0].output_str
            mustach_hash["output_str_right"] = f"weights_{self.name}_{self.idx:02d}"
        elif self.side == 1:
            mustach_hash["output_str_right"] = self.previous_layer[0].output_str
            mustach_hash["output_str_left"] = f"weights_{self.name}_{self.idx:02d}"
        elif self.side == 2:
            mustach_hash["output_str_left"] = self.previous_layer[0].output_str
            mustach_hash["output_str_right"] = self.previous_layer[1].output_str

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                self.local_var,
                self.idx,
                "i")

        with open(self.template_path / "layers" / "template_MatMul.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def matmul_default_implementation(
        old_layer: MatMul,
        version: str,
) -> MatMulDefault:
    """Create a MatMul_Default layer using the attributes of old_layer."""
    return MatMulDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shapes=old_layer.input_shapes,
        weights=old_layer.weights if hasattr(old_layer, "weights") else None,
        side=old_layer.side,
        activation_function=old_layer.activation_function,
    )


matmul_factory.register_implementation(
    None,
    matmul_default_implementation,
)
matmul_factory.register_implementation(
    "default",
    matmul_default_implementation,
)
