"""MaxPooling layer type definition.

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

from .Pooling2D import Pooling2D


class MaxPooling2D(Pooling2D):
    """MaxPooling layer class."""

    def __init__(self: Self, **kwds: int) -> None:
        """Build a MaxPooling layer."""
        super().__init__(**kwds)

        self.name = "MaxPooling2D"
        self.pooling_function = np.amax
        self.local_var = "max"
        self.output_var = self.local_var

    def update_local_vars(self: Self) -> str:
        """Generate local var related code."""
        return f"{self.local_var} = -INFINITY;\n"

    def specific_function(self: Self, index: str, input_of_layer: str) -> str:
        """Generate pooling function code."""
        s = f"if ({input_of_layer}[{index}] > {self.local_var})\n"
        s += f"                                {self.local_var} = {input_of_layer}[{index}];\n"

        return s
