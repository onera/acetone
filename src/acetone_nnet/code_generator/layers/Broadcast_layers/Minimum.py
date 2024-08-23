"""Minimum layer type definition.

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

from .Broadcast import Broadcast


# Return a tensor with where each position (f,i,j) contains the min of all the values at position (f,i,j) in each tensor
class Minimum(Broadcast):
    """Minimum layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build an Minimum layer."""
        super().__init__(**kwargs)
        self.name = "Minimum"
        self.specific_operator = ", "

    def forward_path_layer(self: Self, input_arrays: np.ndarray) -> np.ndarray:
        """Compute output of layer."""
        mini = input_arrays[0]
        for input_array in input_arrays[1:]:
            mini = np.minimum(mini, input_array)
        return self.activation_function.compute(mini)
