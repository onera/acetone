"""ReduceMax layer with default implementation type definition.

*******************************************************************************
* ACETONE: Predictable programprodg framework for ML applications in safety-critical systems
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

from acetone_nnet.generator.layers.reduce.Reduce import Reduce
from acetone_nnet.versioning.layer_factories import reduce_prod_factory

from .ReduceProd import ReduceProd


class ReduceProdDefault(ReduceProd, Reduce):
    """Reduce Prod Default layer with default implementation class."""

    def __init__(self:Self, version:str, **kwargs:Any) -> None:
        """Build a Reduce Prod Layer with default implementation."""
        ReduceProd.__init__(self, **kwargs)
        Reduce.__init__(self, **kwargs)
        self.name = "ReduceProd"
        self.version = version
        self.reduce_func = "Prod"

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        return ReduceProd.forward_path_layer(self, input_array)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str
        return Reduce.generate_inference_code_layer(self, output_str)

def reduce_prod_default_implementation(
        old_layer: ReduceProd,
        version: str,
) -> ReduceProdDefault:
    """Create a Reduce_Prod_Default layer using the parameters of old_layer."""
    return ReduceProdDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        axis=old_layer.axes,
        keepdims=old_layer.keepdims,
        noop_with_empty_axes=old_layer.noop_with_empty_axes,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        activation_function=old_layer.activation_function,
    )

reduce_prod_factory.register_implementation(
    None,
    reduce_prod_default_implementation,
)

reduce_prod_factory.register_implementation(
    "default",
    reduce_prod_default_implementation,
)
