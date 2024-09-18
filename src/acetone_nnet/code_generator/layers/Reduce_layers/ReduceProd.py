"""ReduceProd layer type definition.

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

from .Reduce import Reduce


class ReduceProd(Reduce):
    """ReduceProd layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a ReduceProd layer."""
        super().__init__(**kwargs)
        self.name = "ReduceProd"
        self.reduce_func = "Prod"

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(1, self.input_channels, self.input_height, self.input_width)
        if self.axes == ():
            if self.noop_with_empty_axes:
                return input_array
            return np.multiply.reduce(input_array, axis=None, keepdims=self.keepdims)
        return np.multiply.reduce(input_array, axis=self.axes, keepdims=self.keepdims)
