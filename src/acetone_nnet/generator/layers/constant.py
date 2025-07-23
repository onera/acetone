"""Constant Tensor Layer definition.

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

from typing import Any

import pystache
from traits.api import HasTraits, Instance, Int, Property, provides
from typing_extensions import Self

from acetone_nnet.ir import Operation, Tensor, TensorSpec, layer
from acetone_nnet.versioning.layer_factories import constant_factory


@layer
@provides(Operation)
class ConstantLayer(HasTraits):
    """Constant tensor layer class."""

    #: Constant tensor value
    weights = Instance(Tensor)

    #: Output tensor size, used as redundant check
    size = Property(Int())

    def _get_size(self) -> int:
        return self._infer_shape().size

    def _set_size(self, size: int) -> None:
        if size != self.size:
            msg = f"Provided size {size:d} should be equal to expected {self.size:d}"
            raise ValueError(msg)

    def __init__(
        self: Self,
        original_name: str,
        size: int,
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.name = "Input_layer"
        if original_name == "":
            self.original_name = f"{self.name}_{self.idx}"
        else:
            self.original_name = original_name

    def __call__(self) -> Tensor:
        return self.weights

    def _infer_shape(self) -> TensorSpec:
        return self.weights

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        raise NotImplementedError


class ConstantLayerDefault(ConstantLayer):
    """Constant layer with default implementation."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Constant Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {
            "name": self.name,
            "original_name": self.original_name,
            "idx": f"{self.idx:02d}",
            "weights_var": f"weights_{self.name}_{self.idx:02d}",
            "road": self.path,
            "size": self.size,
        }

        with open(
            self.template_path / "layers" / "template_constant.c.tpl",
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def constant_default_implementation(
    original: ConstantLayer,
    version: str,
) -> ConstantLayerDefault:
    """Create a InputLayer_Default layer using the attributes of old_layer."""
    return ConstantLayerDefault(
        version=version,
        idx=original.idx,
        name=original.name,
        original_name=original.original_name,
        size=original.size,
        weights=original.weights,
    )


constant_factory.register_implementation(
    None,
    constant_default_implementation,
)
constant_factory.register_implementation(
    "default",
    constant_default_implementation,
)

if __name__ == "__main__":
    pass
