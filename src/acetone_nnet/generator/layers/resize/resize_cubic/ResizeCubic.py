"""Cubic Resize base type definition.

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


# The Cubic mode of the Resize layers
# Use a (bi)cubic interpolation to find the new value
class ResizeCubic(Resize):
    """ResizeCubic layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a ResizeCubic layer."""
        super().__init__(**kwargs)
        self.name = "ResizeCubic"
        self.mode = "cubic"
        self.template_dict = {"1D": self.template_path / "layers" / "Resize" / "template_ResizeCubic1D.c.tpl",
                              "2D": self.template_path / "layers" / "Resize" / "template_ResizeCubic2D.c.tpl"}

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if type(self.cubic_coeff_a) is not float and type(self.cubic_coeff_a) is not int:
            msg = "Error: cubic coeff a type in Resize Cubic (must be int or float)"
            raise TypeError(msg)

    def cubic_interpolation_1_d(
            self: Self,
            input_array: np.ndarray,
            f: int,
            x: int,
            y: int,
            s: float,
    ) -> float:
        """Compute the 1D cubic interpolation."""
        col_index = max(0, min(self.input_width - 1, y))
        f_1 = input_array[f, max(0, min(self.input_height - 1, x - 1)), col_index]
        f0 = input_array[f, max(0, min(self.input_height - 1, x)), col_index]
        f1 = input_array[f, max(0, min(self.input_height - 1, x + 1)), col_index]
        f2 = input_array[f, max(0, min(self.input_height - 1, x + 2)), col_index]

        coeff1 = (((self.cubic_coeff_a * (s + 1) - 5 * self.cubic_coeff_a) * (s + 1)
                   + 8 * self.cubic_coeff_a) * (s + 1) - 4 * self.cubic_coeff_a)
        coeff2 = (((self.cubic_coeff_a + 2) * s - (self.cubic_coeff_a + 3)) * s * s + 1)
        coeff3 = (((self.cubic_coeff_a + 2) * (1 - s)
                   - (self.cubic_coeff_a + 3)) * (1 - s) * (1 - s) + 1)
        coeff4 = (((self.cubic_coeff_a * (2 - s) - 5 * self.cubic_coeff_a) * (2 - s)
                   + 8 * self.cubic_coeff_a) * (2 - s) - 4 * self.cubic_coeff_a)

        return f_1 * coeff1 + f0 * coeff2 + f1 * coeff3 + f2 * coeff4

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
                    x0 = math.floor(x)
                    y = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](j, 3)
                    y0 = math.floor(y)

                    s = y - y0
                    coeff1 = (((self.cubic_coeff_a * (s + 1) - 5 * self.cubic_coeff_a) * (s + 1)
                               + 8 * self.cubic_coeff_a) * (s + 1) - 4 * self.cubic_coeff_a)
                    coeff2 = (((self.cubic_coeff_a + 2) * s - (self.cubic_coeff_a + 3)) * s * s + 1)
                    coeff3 = (((self.cubic_coeff_a + 2) * (1 - s) - (self.cubic_coeff_a + 3)) * (1 - s) * (1 - s) + 1)
                    coeff4 = (((self.cubic_coeff_a * (2 - s) - 5 * self.cubic_coeff_a) * (2 - s)
                               + 8 * self.cubic_coeff_a) * (2 - s) - 4 * self.cubic_coeff_a)

                    output[f, i, j] = (coeff1 * self.cubic_interpolation_1_d(input_array, f, x0, y0 - 1, x - x0)
                                       + coeff2 * self.cubic_interpolation_1_d(input_array, f, x0, y0, x - x0)
                                       + coeff3 * self.cubic_interpolation_1_d(input_array, f, x0, y0 + 1, x - x0)
                                       + coeff4 * self.cubic_interpolation_1_d(input_array, f, x0, y0 + 2, x - x0))
        return output

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
