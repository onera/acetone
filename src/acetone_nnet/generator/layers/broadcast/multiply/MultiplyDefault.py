"""Multiply layer with default implementation type definition.

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

import numpy as np
from typing_extensions import Any, Self

from acetone_nnet.generator.layers.broadcast.Broadcast import Broadcast
from acetone_nnet.versioning.layer_factories import multiply_factory

from .Multiply import Multiply


# Addition of several tensors
class MultiplyDefault(Multiply, Broadcast):
    """Multiply layer with default implementation class."""

    def __init__(self: Self, version:str, **kwargs: Any) -> None:
        """Build a Multiply layer with default implementation."""
        Multiply.__init__(self, **kwargs)
        Broadcast.__init__(self, **kwargs)
        self.name = "Multiply"
        self.specific_operator = "*"
        self.version = version

    def forward_path_layer(self: Self, input_arrays: np.ndarray) -> np.ndarray:
        """Compute output of layer."""
        return Multiply.forward_path_layer(self, input_arrays)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        return Broadcast.generate_inference_code_layer(self)

def multiply_default_implementation(
        old_layer: Multiply,
        version: str,
) -> MultiplyDefault:
    """Create a Multiply_Default layer using the parameters of old_layer."""
    return MultiplyDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shapes=old_layer.input_shapes,
        output_shape=[1, old_layer.output_channels, old_layer.output_height, old_layer.output_width],
        activation_function=old_layer.activation_function,
        constant=old_layer.constant,
    )

multiply_factory.register_implementation(
    None,
    multiply_default_implementation,
)
multiply_factory.register_implementation(
    "default",
    multiply_default_implementation,
)
