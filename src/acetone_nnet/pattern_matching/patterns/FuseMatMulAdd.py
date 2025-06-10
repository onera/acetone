"""Replacing MatMul and Add by Dense pattern definition.

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
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import Linear
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import Add, Dense, MatMul
from acetone_nnet.pattern_matching.Pattern import (
    Pattern,
    update_dict_cst,
    update_indices,
    update_next_layers,
    update_previous_layers,
)
from acetone_nnet.pattern_matching.PatternMatcher import pattern_matcher


class MatMulAddToDense(Pattern):
    """Replacing MatMul and Add by Dense pattern class."""

    def __init__(self) -> None:
        """Pattern instantiation."""
        super().__init__(
            name="MatMul followed by Add becomes Dense",
            pattern="MatMul_{0}(Add_{1}(_)) -> Dense_{2}(_)\n",
            shift=0,
        )

    def is_pattern(self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""
        # Check if the layer is a MatMul
        if not isinstance(layer, MatMul):
            return False
        # Check if the layer only has one parent
        if len(layer.previous_layer) != 1:
            return False
        # Check if the input is 1D
        if layer.input_shapes[1:].count(1) < 2:
            return False
        # Check if there is no activation function between this layer and the next
        if not isinstance(layer.activation_function, Linear):
            return False

        # Check the next layer
        if len(layer.next_layer) != 1:
            return False
        if not isinstance(layer.next_layer[0], Add):
            return False
        if len(layer.next_layer[0].previous_layer) > 2:
            return False
        if layer.next_layer[0].constant is None:
            return False

        return True

    def apply_pattern(
        self: Self,
        index: int,
        layers: list[Layer],
        dict_cst: dict[int, int],
    ) -> tuple[str, int]:
        """Apply the pattern to the layer."""
        matmul: MatMul = layers[index]
        add: Add = matmul.next_layer[0]

        # Retrieve the constant
        weights = matmul.weights
        weights = np.moveaxis(weights, 3, 0)
        weights = np.reshape(weights, (weights.shape[-1], weights.shape[0]))
        bias = add.constant[0, 0, 0, :]

        if matmul.side == 1:
            weights = weights.T

        # Create the new layer
        dense = Dense(idx=matmul.idx, size=add.size, weights=weights, biases=bias,
                      activation_function=add.activation_function)

        # Getting the basic parameters of the layer
        dense.next_layer = add.next_layer
        dense.previous_layer = matmul.previous_layer
        dense.path = matmul.path
        dense.output_str = add.output_str
        dense.sorted = matmul.sorted
        dense.fused_layer = add.fused_layer
        update_dict_cst(add, dense, dict_cst)

        # Updating the parents
        update_previous_layers(matmul,dense)

        # Updating the children
        update_next_layers(add,dense)

        # Updating the list of layers
        layers[index] = dense
        layers.pop(index + 1)
        update_indices(index, layers, 1, dict_cst)

        return self.pattern.format(index, index + 1, index), layers.index(dense)

pattern_matcher.register_pattern(MatMulAddToDense())
