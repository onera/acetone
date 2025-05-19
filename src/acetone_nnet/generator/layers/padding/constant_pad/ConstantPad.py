"""ConstantPad layer type definition.

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

from acetone_nnet.generator.layers.padding.Pad import Pad


# The Constant mode of the Pad layers
# Use a constant to fill paddings
class ConstantPad(Pad):
    """ConstantPad layer class."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Build a ConstantPad layer."""
        super().__init__(**kwargs)
        self.name = "ConstantPad"
        self.mode = "constant"

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""
        input_array = input_array.reshape(self.input_shape[1], self.input_shape[2], self.input_shape[3])
        nb_dim = len(self.pads) // 2
        pad_width = [(self.pads[i], self.pads[i + nb_dim]) for i in
                     range(1, nb_dim)]  # Constructing the pads accordingly to the numpy nomenclature
        print(self.constant_value)
        return self.activation_function.compute(
            np.pad(input_array, pad_width=pad_width, mode="constant", constant_values=self.constant_value),
        )