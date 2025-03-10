"""AveragePooling layer type definition.

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


class AveragePooling2D(Pooling2D):
    """AveragePooling layer class."""

    def __init__(self: Self, **kwds: int) -> None:
        """Build an AveragePooling layer."""
        super().__init__(**kwds)

        self.name = "AveragePooling2D"
        self.pooling_function = np.mean
        self.local_var = "sum"
        self.local_var_2 = "count"
        self.output_var = f"{self.local_var}/{self.local_var_2}"

    def update_local_vars(self: Self) -> str:
        """Generate local var related code."""
        return f"{self.local_var} = 0; {self.local_var_2} = 0;\n"

    def specific_function(self: Self, index: str, input_of_layer: str) -> str:
        """Generate pooling function code."""
        # Computes the average in this subclass AveragePooling2D
        s = f"{self.local_var} += {input_of_layer} [{index}];\n"
        s += f"                            {self.local_var_2}++;\n"

        return s
