"""Layer factories definition.

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

LayerVariant = Callable[[Layer, str], Layer]

class LayerFactory:
    """Build Layer implementation."""

    def __init__(self, layer_name:str) -> None:
        """Build default Layer factory."""
        self.layer_name = layer_name
        self.implementations: dict[str | None, LayerVariant] = {
        }

    @property
    def list_implementations(self) -> list[str]:
        """Return known Layer implementations."""
        return [i for i in self.implementations if i is not None]

    def register_implementation(self, name: str, variant: LayerVariant) -> None:
        """Register a new Layer variant."""
        if name in self.implementations:
            msg = f"{self.layer_name} variant {name} already exists."
            raise KeyError(msg)
        self.implementations[name] = variant
        self.implementations = dict(
            sorted(
                self.implementations.items(),
                key=lambda x: "a" if x[0] is None else x[0]
            )
        )

    def __call__(self, layer: Layer, version: str) -> Layer:
        """Create a Layer implementation layer for the required implementation."""
        if version not in self.implementations:
            msg = f"Unknown {self.layer_name} variant {version}."
            raise KeyError(msg)

        return self.implementations[version](layer, version)

add_factory = LayerFactory("Add")
average_factory = LayerFactory("Average")
average_pooling_factory = LayerFactory("AveragePooling2D")
batch_normalization_factory = LayerFactory("BatchNormalization")
concatenate_factory = LayerFactory("Concatenate")
constant_pad_factory = LayerFactory("ConstantPad")
conv2d_factory = LayerFactory("Conv2D")
dense_factory = LayerFactory("Dense")
divide_factory = LayerFactory("Divide")
edge_pad_factory = LayerFactory("EdgePad")
flatten_factory = LayerFactory("Flatten")
gather_factory = LayerFactory("Gather")
gather_elements_factory = LayerFactory("GatherElements")
gemm_factory = LayerFactory("Gemm")
input_factory = LayerFactory("Input_layer")
matmul_factory = LayerFactory("MatMul")
max_pooling_factory = LayerFactory("MaxPooling2D")
maximum_factory = LayerFactory("Maximum")
minimum_factory = LayerFactory("Minimum")
multiply_factory = LayerFactory("Multiply")
reduce_max_factory = LayerFactory("ReduceMax")
reduce_mean_factory = LayerFactory("ReduceMean")
reduce_min_factory = LayerFactory("ReduceMin")
reduce_prod_factory = LayerFactory("ReduceProd")
reduce_sum_factory = LayerFactory("ReduceSum")
reflect_pad_factory = LayerFactory("ReflectPad")
resize_cubic_factory = LayerFactory("ResizeCubic")
resize_linear_factory = LayerFactory("ResizeLinear")
resize_nearest_factory = LayerFactory("ResizeNearest")
softmax_factory = LayerFactory("Softmax")
subtract_factory = LayerFactory("Subtract")
tile_factory = LayerFactory("Tile")
transpose_factory = LayerFactory("Transpose")
wrap_pad_factory = LayerFactory("WrapPad")

implemented: dict[str, LayerFactory] = {
    "Add":add_factory,
    "Average":average_factory,
    "AveragePooling2D":average_pooling_factory,
    "BatchNormalization": batch_normalization_factory,
    "Concatenate": concatenate_factory,
    "ConstantPad":constant_pad_factory,
    "Conv2D": conv2d_factory,
    "Dense": dense_factory,
    "Divide":divide_factory,
    "EdgePad":edge_pad_factory,
    "Flatten": flatten_factory,
    "Gather": gather_factory,
    "GatherElements": gather_elements_factory,
    "Gemm": gemm_factory,
    "Input_layer": input_factory,
    "MatMul": matmul_factory,
    "MaxPooling2D":max_pooling_factory,
    "Maximum":maximum_factory,
    "Minimum":minimum_factory,
    "Multiply":multiply_factory,
    "ReduceMax":reduce_max_factory,
    "ReduceMean":reduce_mean_factory,
    "ReduceMin":reduce_min_factory,
    "ReduceProd":reduce_prod_factory,
    "ReduceSum":reduce_sum_factory,
    "ReflectPad":reflect_pad_factory,
    "ResizeCubic":resize_cubic_factory,
    "ResizeLinear":resize_linear_factory,
    "ResizeNearest":resize_nearest_factory,
    "Softmax": softmax_factory,
    "Subtract":subtract_factory,
    "Tile": tile_factory,
    "Transpose": transpose_factory,
    "WrapPad":wrap_pad_factory,
}
