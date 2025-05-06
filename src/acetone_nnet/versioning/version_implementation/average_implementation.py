"""Average versioning manager.

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

from acetone_nnet.generator.layers.broadcast import Average

AverageVariant = Callable[[Average, str], Average]


class AverageFactory:
    """Build Average implementation layers."""

    def __init__(self) -> None:
        """Build default Average layer factory."""
        self.implementations: dict[str | None, AverageVariant] = {
        }

    @property
    def list_implementations(self) -> list[str]:
        """Return known Average implementations."""
        return [i for i in self.implementations if i is not None]

    def register_implementation(self, name: str, variant: AverageVariant) -> None:
        """Register a new Average variant."""
        if name in self.implementations:
            msg = f"Average variant {name} already exists."
            raise KeyError(msg)
        self.implementations[name] = variant

    def __call__(self, layer: Average, version: str) -> Average:
        """Create a Average implementation layer for the required implementation."""
        if version not in self.implementations:
            msg = f"Unknown Average variant {version}."
            raise KeyError(msg)

        return self.implementations[version](layer, version)


average_factory = AverageFactory()
