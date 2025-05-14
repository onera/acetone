"""BatchNormalization layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import batch_normalization_factory

from .BatchNormalization import BatchNormalization


class BatchNormalizationDefault(BatchNormalization):
    """BatchNormalization layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a BatchNormalization layer with default implementation."""
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
        mustach_hash["path"] = self.path

        if self.activation_function.name != "linear":
            mustach_hash["activation_function"] = self.activation_function.write_activation_str(
                f"output_{self.path}[k + {self.output_height * self.output_width}*f]")

        mustach_hash["input_channels"] = self.output_channels
        mustach_hash["channel_size"] = self.output_height * self.output_width
        mustach_hash["epsilon"] = self.epsilon

        with open(self.template_path / "layers" / "template_BatchNormalization.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def batch_normalization_default_implementation(
        old_layer: BatchNormalization,
        version, str,
) -> BatchNormalizationDefault:
    """Create a BatchNormalization_Default layer using the attributes of old_layer."""
    return BatchNormalizationDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        epsilon=old_layer.epsilon,
        scale=old_layer.scale,
        biases=old_layer.biases,
        mean=old_layer.mean,
        var=old_layer.var,
        activation_function=old_layer.activation_function,
    )


batch_normalization_factory.register_implementation(
    None,
    batch_normalization_default_implementation,
)
batch_normalization_factory.register_implementation(
    "default",
    batch_normalization_default_implementation,
)
