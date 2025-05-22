"""Replacing Sigmoid and Mul by Silu pattern definition.

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

from acetone_nnet.generator.activation_functions import Linear, Sigmoid, Silu
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import ActivationLayer, Multiply
from acetone_nnet.pattern_matching.Pattern import (
    Pattern,
    update_indices,
    update_next_layers,
)
from acetone_nnet.pattern_matching.PatternMatcher import pattern_matcher


class SigmoidMultiplyToSilu(Pattern):
    """Replacing Sigmoid and Mul by Silu pattern class."""

    def __init__(self):
        """Pattern instantiation."""
        super().__init__(
            name=" Multiply Sigmoid output by input becomes Silu",
            pattern="Layer_{0}(Sigmoid_{1}(Multiply_{2}(_)), Multiply_{2}(_)) -> Layer_{0}(Silu{3}(_))\n",
            shift=1,
        )

    def is_pattern(self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""
        # Check if the layer is a Sigmoid
        if not isinstance(layer, ActivationLayer):
            return False
        if layer.name != "Sigmoid":
            return False
        if not isinstance(layer.activation_function, Sigmoid):
            return False

        # Check if:
        #   - there is only one previous layer
        #   - the previous_layer has no activation function
        #   - there is only one next layer
        #   - the previous layer has an attribute activation_function
        #   - the previous_layer has exactly two children
        #   - the next layer is a Multiply
        #   - the next layer has exactly two parents
        #   - the next layer is in previous_layer.next_layer
        if len(layer.previous_layer) != 1:
            return False
        if not isinstance(layer.previous_layer[0].activation_function, Linear):
            return False
        if len(layer.next_layer) != 1:
            return False
        if not hasattr(layer.previous_layer[0], "activation_function"):
            return False
        if len(layer.previous_layer[0].next_layer) != 2:
            return False
        if not isinstance(layer.next_layer[0], Multiply):
            return False
        if len(layer.next_layer[0].previous_layer) != 2:
            return False
        if layer.next_layer[0] not in layer.previous_layer[0].next_layer:
            return False

        return True

    def apply_pattern(
        self: Self,
        index: int,
        layers: list[Layer],
        dict_cst: dict[int, int],
    ) -> str:
        """Apply the pattern to the layer."""
        sigmoid: ActivationLayer = layers[index]
        layer: Layer = sigmoid.previous_layer[0]
        multiply: Multiply = sigmoid.next_layer[0]

        # Create the new layer
        silu = ActivationLayer(
            idx=sigmoid.idx,
            size=sigmoid.size,
            activation_function=Silu(),
        )

        # Getting the basic parameters of the layer
        silu.next_layer = multiply.next_layer
        silu.previous_layer = [layer]
        silu.path = multiply.path
        silu.output_str = multiply.output_str
        silu.sorted = multiply.sorted
        silu.fused_layer = multiply.fused_layer

        # Updating the parents
        layer.next_layer.remove(multiply)
        layer.next_layer.remove(sigmoid)
        layer.next_layer.append(silu)

        # Updating the children
        update_next_layers(multiply, silu)

        # Updating the list of layers
        layers[index] = silu
        layers.remove(multiply)
        update_indices(index, layers, 1)

        return self.pattern.format(layer.idx, sigmoid.idx, multiply.idx, silu.idx)

pattern_matcher.register_pattern(SigmoidMultiplyToSilu())
