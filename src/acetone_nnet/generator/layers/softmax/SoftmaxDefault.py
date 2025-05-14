"""Softmax Layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import softmax_factory

from .Softmax import Softmax


class SoftmaxDefault(Softmax):
    """Softmax layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Softmax Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["road"] = self.path
        mustach_hash["output_str"] = output_str

        mustach_hash["1D"] = self.one_dimension

        if self.one_dimension:
            mustach_hash["size"] = self.size
        else:
            mustach_hash["output_channels"] = self.output_channels
            mustach_hash["output_height"] = self.output_height
            mustach_hash["output_width"] = self.output_width

            if self.axis == 1:
                mustach_hash["sum_dimension_1"] = self.output_height
                mustach_hash["sum_dimension_2"] = self.output_width
                mustach_hash["reduced_dimension"] = self.output_channels
                mustach_hash["reduced_position_1"] = f"i + {self.output_width}*f"
                mustach_hash["reduced_position_2"] = f"i + {self.output_width}*(f + {self.output_height}*j)"
                mustach_hash["softmax_indice"] = f"j + {self.output_width}*i"

            elif self.axis == 2:
                mustach_hash["sum_dimension_1"] = self.output_channels
                mustach_hash["sum_dimension_2"] = self.output_width
                mustach_hash["reduced_dimension"] = self.output_height
                mustach_hash["reduced_position_1"] = f"i + {self.output_width}*f"
                mustach_hash["reduced_position_2"] = f"i + {self.output_width}*(j + {self.output_height}*f)"
                mustach_hash["softmax_indice"] = f"j + {self.output_width}*f"

            elif self.axis == 3:
                mustach_hash["sum_dimension_1"] = self.output_channels
                mustach_hash["sum_dimension_2"] = self.output_height
                mustach_hash["reduced_dimension"] = self.output_width
                mustach_hash["reduced_position_1"] = f"i + {self.output_height}*f"
                mustach_hash["reduced_position_2"] = f"j + {self.output_width}*(i + {self.output_height}*f)"
                mustach_hash["softmax_indice"] = f"i + {self.output_height}*f"

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                f"output_{self.path}[j]",
                self.idx,
                "j")

        with open(self.template_path / "layers" / "template_Softmax.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def softmax_default_implementation(
        old_layer: Softmax,
        version: str,
) -> SoftmaxDefault:
    """Create a Softmax_Default layer using the attributes of old_layer."""
    if old_layer.one_dimension:
        output_shape = [1, 1, 1, old_layer.size]
    else:
        output_shape = [1, old_layer.output_channels, old_layer.output_height, old_layer.output_width]
    return SoftmaxDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        output_shape=output_shape,
        axis=old_layer.axis,
    )

softmax_factory.register_implementation(
    None,
    softmax_default_implementation,
)
softmax_factory.register_implementation(
    "default",
    softmax_default_implementation,
)
