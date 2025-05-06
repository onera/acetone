"""Generic layer versioning manager.

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
from collections.abc import Callable

from acetone_nnet.generator import Layer
from acetone_nnet.versioning.version_implementation.add_implementation import (
    add_factory,
)
from acetone_nnet.versioning.version_implementation.average_implementation import (
    average_factory,
)
from acetone_nnet.versioning.version_implementation.batch_normalization_implementation import (
    batch_normalization_factory,
)
from acetone_nnet.versioning.version_implementation.concatenate_implementation import (
    concatenate_factory,
)
from acetone_nnet.versioning.version_implementation.conv_implementation import (
    conv2d_factory,
)
from acetone_nnet.versioning.version_implementation.dense_implementation import (
    dense_factory,
)
from acetone_nnet.versioning.version_implementation.divide_implementation import (
    divide_factory,
)
from acetone_nnet.versioning.version_implementation.flatten_implementation import (
    flatten_factory,
)
from acetone_nnet.versioning.version_implementation.gather_elements_implementation import (
    gather_elements_factory,
)
from acetone_nnet.versioning.version_implementation.gather_implementation import (
    gather_factory,
)
from acetone_nnet.versioning.version_implementation.gemm_implementation import (
    gemm_factory,
)
from acetone_nnet.versioning.version_implementation.input_implementation import (
    input_factory,
)
from acetone_nnet.versioning.version_implementation.matmul_implementation import (
    matmul_factory,
)
from acetone_nnet.versioning.version_implementation.maximum_implementation import (
    maximum_factory,
)
from acetone_nnet.versioning.version_implementation.minimum_implementation import (
    minimum_factory,
)
from acetone_nnet.versioning.version_implementation.multiply_implementation import (
    multiply_factory,
)
from acetone_nnet.versioning.version_implementation.reduce_max_implementation import (
    reduce_max_factory,
)
from acetone_nnet.versioning.version_implementation.reduce_mean_implementation import (
    reduce_mean_factory,
)
from acetone_nnet.versioning.version_implementation.reduce_min_implementation import (
    reduce_min_factory,
)
from acetone_nnet.versioning.version_implementation.reduce_prod_implementation import (
    reduce_prod_factory,
)
from acetone_nnet.versioning.version_implementation.reduce_sum_implementation import (
    reduce_sum_factory,
)
from acetone_nnet.versioning.version_implementation.resize_cubic_implementation import (
    resize_cubic_factory,
)
from acetone_nnet.versioning.version_implementation.resize_linear_implementation import (
    resize_linear_factory,
)
from acetone_nnet.versioning.version_implementation.resize_nearest_implementation import (
    resize_nearest_factory,
)
from acetone_nnet.versioning.version_implementation.softmax_implementation import (
    softmax_factory,
)
from acetone_nnet.versioning.version_implementation.subtract_implementation import (
    subtract_factory,
)
from acetone_nnet.versioning.version_implementation.tile_implementation import (
    tile_factory,
)
from acetone_nnet.versioning.version_implementation.transpose_implementation import (
    transpose_factory,
)

LayerFactory = Callable[[Layer, str], Layer]


def versioning(
        layers: list[Layer],
        version: dict[int, str],
) -> list[Layer]:
    """Check layers and change the layer version if needed."""
    implemented: dict[str, LayerFactory] = {
        "Conv2D": conv2d_factory,
        "BatchNormalization": batch_normalization_factory,
        "Concatenate": concatenate_factory,
        "Dense": dense_factory,
        "Flatten": flatten_factory,
        "Gather": gather_factory,
        "GatherElements": gather_elements_factory,
        "Gemm": gemm_factory,
        "Input_layer": input_factory,
        "MatMul": matmul_factory,
        "Softmax": softmax_factory,
        "Tile": tile_factory,
        "Transpose": transpose_factory,
        "ReduceMax":reduce_max_factory,
        "ReduceMean":reduce_mean_factory,
        "ReduceMin":reduce_min_factory,
        "ReduceProd":reduce_prod_factory,
        "ReduceSum":reduce_sum_factory,
        "ResizeCubic":resize_cubic_factory,
        "ResizeLinear":resize_linear_factory,
        "ResizeNearest":resize_nearest_factory,
        "Add":add_factory,
        "Average":average_factory,
        "Divide":divide_factory,
        "Maximum":maximum_factory,
        "Minimum":minimum_factory,
        "Multiply":multiply_factory,
        "Subtract":subtract_factory,
    }

    keys = list(version.keys())
    for idx in keys:
        for j in range(len(layers)):
            if layers[j].idx == idx:
                layer = layers[j]

                layer = implemented[layer.name](layer, version[idx])
                layer.path = layers[j].path
                layer.next_layer = layers[j].next_layer
                layer.previous_layer = layers[j].previous_layer
                layer.sorted = layers[j].sorted
                layer.output_str = layers[j].output_str
                layer.fused_layer = layers[j].fused_layer

                layers[j] = layer

    return layers
