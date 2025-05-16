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

from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import BatchNormalization, Conv2D
from acetone_nnet.pattern_matching.Pattern import Pattern, update_indices


class FuseConvBatchNorm(Pattern):
    """Fusing Conv2D and BatchNormalization pattern class."""

    def __init__(self):
        """Fusing Conv2D and BatchNormalization pattern instantiation."""
        super().__init__(
            name="Conv2D followed by BatchNormalization becomes Conv2D_new",
            pattern="Conv2D_{0}(BatchNormalization_{1}(_)) -> Conv2D_new_{0}(_)\n"
        )

    def is_pattern(self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""
        # Checking if the layer is a BatchNormalization
        if not isinstance(layer, BatchNormalization):
            return False

        # Checking if the previous layer exists, is a Conv2D and has only one child
        if len(layer.previous_layer) != 1:
            return False
        if not isinstance(layer.previous_layer[0], Conv2D):
            return False
        if len(layer.previous_layer[0].next_layer) != 1:
            return False

        return True

    def apply_pattern(self, index: int, layers: list[Layer]) -> str:
        """Apply the pattern to the layer."""
        batch_norm: BatchNormalization = layers[index]
        conv: Conv2D = layers[index - 1]

        # Retrieve the constant
        weights: np.array = conv.weights
        biases = conv.biases

        scale = batch_norm.scale
        bias = batch_norm.biases
        mean = batch_norm.mean
        var = batch_norm.var
        epsilon = batch_norm.epsilon

        for z in range(len(weights[0, 0, 0, :])):
            alpha = scale[z] / np.sqrt(var[z] + epsilon)
            B = bias[z] - (mean[z] * alpha)
            weights[:, :, :, z] = alpha * weights[:, :, :, z]
            biases[z] = alpha * biases[z] + B

        conv.weights = weights
        conv.biases = biases

        for next_layer in batch_norm.next_layer:
            next_layer.previous_layer.remove(batch_norm)
            next_layer.previous_layer.append(conv)

        conv.next_layer.remove(batch_norm)

        # Updating the list of layers
        layers.pop(index)
        update_indices(index - 1, layers, 1)

        return self.pattern.format(index - 1, index, index - 1)