"""ReduceProd versioning manager.

*******************************************************************************
* ACETONE: Predictable programprodg framework for ML applications in safety-critical systems
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

from acetone_nnet.generator.layers.reduce import ReduceProd

ReduceProdVariant = Callable[[ReduceProd, str], ReduceProd]


class ReduceProdFactory:
    """Build ReduceProd implementation layers."""

    def __init__(self) -> None:
        """Build default ReduceProd layer factory."""
        self.implementations: dict[str | None, ReduceProdVariant] = {
        }

    @property
    def list_implementations(self) -> list[str]:
        """Return known ReduceProd implementations."""
        return [i for i in self.implementations if i is not None]

    def register_implementation(self, name: str, variant: ReduceProdVariant) -> None:
        """Register a new ReduceProd variant."""
        if name in self.implementations:
            msg = f"ReduceProd variant {name} already exists."
            raise KeyError(msg)
        self.implementations[name] = variant

    def __call__(self, layer: ReduceProd, version: str) -> ReduceProd:
        """Create a ReduceProd implementation layer for the required implementation."""
        if version not in self.implementations:
            msg = f"Unknown ReduceProd variant {version}."
            raise KeyError(msg)

        return self.implementations[version](layer, version)


reduce_prod_factory = ReduceProdFactory()
