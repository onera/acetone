"""Linear Resize base type definition.

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
from typing_extensions import Self

from acetone_nnet.generator.layers.resize.Resize import Resize


# The value in the output tensor are found thanks to a (bi)linear interpolation
class ResizeLinear(Resize):
    """ResizeLinear layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a ResizeLinear layer."""
        super().__init__(**kwargs)
        self.name = "ResizeLinear"
        self.mode = "linear"
        self.template_dict = {"1D": self.template_path / "layers" / "Resize" / "template_ResizeLinear1D.c.tpl",
                              "2D": self.template_path / "layers" / "Resize" / "template_ResizeLinear2D.c.tpl"}

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
                    x = np.clip(x, 0, self.input_height - 1)
                    y = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](j, 3)
                    y = np.clip(y, 0, self.input_width - 1)

                    x0, x1 = math.floor(x), math.floor(x) + 1
                    y0, y1 = math.floor(y), math.floor(y) + 1

                    f11 = input_array[f, x0, y0]
                    f21 = 0 if x1 >= self.input_height else input_array[f, x1, y0]
                    if x1 >= self.input_height or y1 >= self.input_width:
                        f22 = 0
                    else:
                        f22 = input_array[f, x1, y1]
                    f12 = 0 if y1 >= self.input_width else input_array[f, x0, y1]

                    output[f, i, j] = ((f11 * (x1 - x) * (y1 - y) + f21 * (x - x0) * (y1 - y)
                                        + f12 * (x1 - x) * (y - y0) + f22 * (x - x0) * (y - y0))
                                       / ((x1 - x0) * (y1 - y0)))

        return output
