"""Activation Layers type definition.

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

from abc import abstractmethod

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.Layer import Layer


class ActivationLayer(Layer):
    """Abstract class for activation layers."""

    def __init__(self: Self, idx: int, op_type: str, size: int, **kwargs) -> None:  # noqa: ANN003
        """Initiate the class."""
        super().__init__()

        self.idx = idx
        self.op_type = op_type
        self.size = size
        self.kwargs = kwargs

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Activation Layer (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Activation Layer (size must be int)"
            msg += "\n"
        if type(self.op_type) is not str:
            msg += "Error: op_type type in Activation Layer (op_type must be str)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.op_type not in [
            "sigmoid",
            "relu",
            "leakyrelu",
            "hyperb_tan",
            "linear",
            "Exponential",
            "Logarithm",
            "Clip",
        ]:
            msg += f"Error: op_type value in Activation Layer ({self.conv_algorithm})"
            msg += "\n"
        if msg:
            raise ValueError(msg)

    @abstractmethod
    def forward_path_layer(
            self: Self,
            inputs: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """Compute output of layer."""

    @abstractmethod
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
