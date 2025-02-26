"""Conv versioning manager.

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

from acetone_nnet.generator.layers.convolution import Conv2D

Conv2DVariant = Callable[[Conv2D, str], Conv2D]


class Conv2DFactory:
    """Build Conv2D implementation layers."""

    def __init__(self) -> None:
        """Build default convolution layer factory."""
        self.implementations: dict[str | None, Conv2DVariant] = {
        }

    @property
    def list_implementations(self) -> list[str]:
        """Return known convolution implementations."""
        return [i for i in self.implementations if i is not None]

    def register_implementation(self, name: str, variant: Conv2DVariant) -> None:
        """Register a new Conv2D variant."""
        # if name in self.implementations:
        #     msg = f"Convolution variant {name} already exists."
        #     raise KeyError(msg)
        self.implementations[name] = variant

    def __call__(self, layer: Conv2D, version: str) -> Conv2D:
        """Create a Convolution implementation layer for the required implementation."""
        if version not in self.implementations:
            msg = f"Unknown convolution variant {version}."
            raise KeyError(msg)

        return self.implementations[version](layer, version)


conv2d_factory = Conv2DFactory()
