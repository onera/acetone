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
