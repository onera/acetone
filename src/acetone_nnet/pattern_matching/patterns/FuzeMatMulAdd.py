"""Fusing Conv2D and BatchNormalization pattern definition.

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

from acetone_nnet.generator.activation_functions import Linear
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import Add, Dense, MatMul
from acetone_nnet.pattern_matching.Pattern import Pattern, update_indices


class MatMulAddToDense(Pattern):
    """Fusing MatMul and Add to Dense pattern class."""

    def __init__(self):
        """Fusing MatMul and Add to Dense pattern instantiation."""
        super().__init__(
            name="MatMul followed by Add becomes Dense",
            pattern="MatMul_{0}(Add_{1}(_)) -> Dense_{2}(_)\n"
        )

    def is_pattern(self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""
        # Check if the layer is a MatMul
        if not isinstance(layer, MatMul):
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

    def apply_pattern(self, index: int, layers: list[Layer]) -> str:
        """Apply the pattern to the layer."""
        matmul: MatMul = layers[index]
        add: Add = matmul.next_layer[0]

        # Retrieve the constant
        weights = matmul.weights
        bias = add.constant[0, 0, 0, :]

        if matmul.side == 1:
            weights = np.moveaxis(weights, 3, 0)
            weights = np.reshape(weights, (weights.shape[-1], weights.shape[0]))
            weights = weights.T
        else:
            weights = weights[0, 0, :, :]

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

        # Updating the parents
        for prev in dense.previous_layer:
            prev.next_layer.remove(matmul)
            prev.next_layer.append(dense)

        # Updating the children
        for next_layer in dense.next_layer:
            next_layer.previous_layer.remove(add)
            next_layer.previous_layer.append(dense)

        # Updating the list of layers
        layers[index] = dense
        layers.pop(index + 1)
        update_indices(index, layers, 1)

        return self.pattern.format(index, index + 1, index)
