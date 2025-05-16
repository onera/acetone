"""Base Pattern type definition.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Self

from acetone_nnet.generator.Layer import Layer


@dataclass
class Pattern(ABC):
    """Base class for patterns."""

    def __init__(self: Self, name: str, pattern: str) -> None:
        """Build a non-specific pattern."""
        super().__init__()
        self.name = name
        self.pattern = pattern

    @abstractmethod
    def is_pattern(self: Self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""

    @abstractmethod
    def apply_pattern(self: Self, index: int, layers: list[Layer]) -> str:
        """Apply the pattern to the layer."""
