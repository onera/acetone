"""MatMul Layer type definition.

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

import logging

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.ir import Layer
from acetone_nnet.quantize import qform


class MatMul(Layer):
    """MatMul layer class."""

    def __init__(
        self: Self,
        original_name: str,
        idx: int,
        size: int,
        input_shapes: list,
        activation_function: ActivationFunctions,
    ) -> None:
        """Build a MatMul layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "MatMul"
        if original_name == "":
            self.original_name = f"{self.name}_{self.idx}"
        else:
            self.original_name = original_name
        self.activation_function = activation_function
        self.local_var = "dotproduct"
        self.input_shapes = input_shapes

        self.output_channels = self.input_shapes[0][1]
        self.output_height = self.input_shapes[0][-2]
        self.output_width = self.input_shapes[1][-1]
        self.shared_dimension = self.input_shapes[0][-1]

        ### Checking value consistency ###
        msg = ""
        if self.input_shapes[1][-2] != self.input_shapes[0][-1]:
            msg = f"Error: non consistency between weight shape and input shape in MatMul ({self.input_shapes[1][-2]}!={self.input_shapes[0][-1]})"
            msg += "\n"
        if (
            self.size
            != self.input_shapes[1][-1]
            * self.input_shapes[0][-2]
            * self.output_channels
        ):
            msg = f"Error: size value in MatMul ({self.size} !={self.input_shapes[1][-1] * self.input_shapes[0][-2]})"
            msg += "\n"
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    def compute_post_shift(self):
        """Q compute the rescaling factor"""
        if hasattr(self, "qparam"):
            (_, mparam) = qform.parse_q_format(self.qparam)
            (_, min) = qform.parse_q_format(self.qin)
            (_, mout) = qform.parse_q_format(self.qout)
            qpost_shift = min + mparam - mout
            if qpost_shift < 0:
                logging.warning(
                    f"MatMul {self} qpost_shift invalid {qpost_shift}, take 0",
                )
                qpost_shift = 0
            return qpost_shift

    def forward_path_layer(
        self: Self,
        input_array: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """Compute output of layer."""
        out = np.array([])
        input_1 = input_array[0].reshape(self.input_shapes[0])
        input_2 = input_array[1].reshape(self.input_shapes[1])
        quantized = any(np.issubdtype(i.dtype, np.integer) for i in input_array)
        if quantized:
            qtype = (
                input_1.dtype
                if np.issubdtype(input_1.dtype, np.integer)
                else input_2.dtype
            )
            out = np.matmul(input_1, input_2, dtype=self.temp_pydtype)
            out = np.right_shift(out, self.compute_post_shift()).astype(qtype)
        else:
            out = np.matmul(input_1, input_2)
        out = self.activation_function.compute(out)
        return out  # Case should not be happening
