"""Nearest Resize base type definition.

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

import math

import numpy as np
from acetone_nnet.generator.layers.resize.Resize import Resize
from typing_extensions import Self


# The mode Nearest of the Resize layers.
# The value in the new tensor is found by applying an rounding operation
class ResizeNearest(Resize):
    """ResizeNearest layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a ResizeNearest layer."""
        super().__init__(**kwargs)
        self.name = "ResizeNearest"
        self.mode = "nearest"

        self.nearest_mode_implem_mapping = {"round_prefer_floor": self.round_prefer_floor_implem,
                                            "round_prefer_ceil": self.round_prefer_ceil_implem,
                                            "floor": math.floor,
                                            "ceil": math.ceil}

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if type(self.nearest_mode) is not str:
            msg = "Error: nearest mode in Resize Nearest (must be string)"
            raise TypeError(msg)

        ### Checking value consistency ###
        if self.nearest_mode not in ["round_prefer_floor",
                                     "round_prefer_ceil",
                                     "floor",
                                     "ceil"]:
            msg = f"Error: nearest mode value in Resize Nearest ({self.nearest_mode})"
            raise ValueError(msg)

    def round_prefer_floor_implem(self: Self, x: int) -> float:
        """Compute round prefer floor code."""
        return math.floor(math.ceil(2 * x) / 2)

    def round_prefer_ceil(self: Self, x: str, y: str) -> str:
        """Generate round prefer ceil code."""
        return f"{x} = ceil(floor(2*{y})/2);"

    def round_prefer_ceil_implem(self: Self, x: int) -> float:
        """Compute round prefer ceil code."""
        return math.ceil(math.floor(2 * x) / 2)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(self.input_channels, self.input_height, self.input_width)
        output = np.zeros((self.output_channels, self.output_height, self.output_width))
        for f in range(self.output_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    x = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](i, 2)
                    x0 = self.nearest_mode_implem_mapping[self.nearest_mode](x)
                    y = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](j, 3)
                    y0 = self.nearest_mode_implem_mapping[self.nearest_mode](y)

                    output[f, i, j] = input_array[f, x0, y0]
        return output
