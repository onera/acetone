"""Fusing Activation layer to the previous one pattern definition.

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
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import Linear
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import ActivationLayer
from acetone_nnet.pattern_matching.Pattern import (
    Pattern,
    update_dict_cst,
    update_indices,
    update_next_layers,
)
from acetone_nnet.pattern_matching.PatternMatcher import pattern_matcher


class FuseActivationLayer(Pattern):
    """Fusing Activation layer to the previous one pattern class."""

    def __init__(self):
        """Pattern instantiation."""
        super().__init__(
            name="Fusing Activation layer to the previous one",
            pattern="{4}_{0}({3}_{1}(_)) -> {4}_{2}(_)\n",
            shift=1,
        )

    def is_pattern(self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""
        # Check if the layer is an Activation layer
        if not isinstance(layer, ActivationLayer):
            return False

        #Check if the previous layer:
        #   - exists
        #   - is not an activation layer
        #   - can be fused with an activation function
        #   - only has one child
        if len(layer.previous_layer) != 1:
            return False
        if isinstance(layer.previous_layer[0], ActivationLayer):
            return False
        if not hasattr(layer.previous_layer[0], "activation_function"):
            return False
        if not isinstance(layer.previous_layer[0].activation_function, Linear):
            return False
        if len(layer.previous_layer[0].next_layer) != 1:
            return False

        return True

    def apply_pattern(
        self: Self,
        index: int,
        layers: list[Layer],
        dict_cst: dict[int, int],
    ) -> tuple[str, int]:
        """Apply the pattern to the layer."""
        activation_layer: ActivationLayer = layers[index]
        layer: Layer = activation_layer.previous_layer[0]

        layer.activation_function = activation_layer.activation_function
        update_dict_cst(activation_layer,layer,dict_cst)

        layer.next_layer.remove(activation_layer)
        update_next_layers(activation_layer, layer)

        # Updating the list of layers
        layers.pop(index)
        update_indices(index - 1, layers, 1, dict_cst)

        return self.pattern.format(
            layer.idx,
            activation_layer.idx,
            layer.idx,
            activation_layer.name,
            layer.name,
        ), layers.index(layer)

pattern_matcher.register_pattern(FuseActivationLayer())
