"""Normalization tool for NNet models.

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

from abc import ABC
from pathlib import Path

import numpy as np
import pystache
from acetone_nnet import templates
from typing_extensions import Self


class Normalizer(ABC):
    """Input and output normalization for NNet file."""

    def __init__(
            self: Self,
            input_size: int,
            output_size: int,
            mins: list[float],
            maxes: list[float],
            means: list[float],
            ranges: list[float],
    ) -> None:
        """Initialize Normalizer class."""
        self.input_size = input_size
        self.output_size = output_size
        self.mins = mins
        self.maxes = maxes
        self.means = means
        self.ranges = ranges
        self.template_path = Path(templates.__file__).parent
        super().__init__()

    @staticmethod
    def array_to_str(
            array: list,
    ) -> str:
        """Transform an array in a string."""
        s = "{"
        for element in array:
            s += str(element) + ", "
        return s[:-2] + "}"

    def write_pre_processing(self: Self) -> str:
        """Generate the pre-processing string of the C code."""
        template_pre_processing = (
                self.template_path / "normalization" / "template_pre_processing.c.tpl"
        )
        with Path.open(template_pre_processing) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, {"input_size": self.input_size})

    def write_post_processing(self: Self) -> str:
        """Generate the post-processing string of the C code."""
        template_post_processing = (
                self.template_path / "normalization" / "template_post_processing.c.tpl"
        )
        with Path.open(template_post_processing) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, {"output_size": self.output_size})

    def write_normalization_cst_in_header_file(self: Self) -> str:
        """Generate the constant header string of the C code."""
        mustach_hash = {"input_size": self.input_size}
        template_cst_header = (
                self.template_path / "normalization" /
                "template_normalization_cst_in_header_file.c.tpl"
        )

        with Path.open(template_cst_header) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_normalization_cst_in_globalvars_file(self: Self) -> Self:
        """Generate the global_var string of the C code."""
        mustach_hash = {
            "input_size": self.input_size,
            "input_min": self.array_to_str(self.mins),
            "input_max": self.array_to_str(self.maxes),
            "input_mean": self.array_to_str(self.means[:-1]),
            "input_range": self.array_to_str(self.ranges[:-1]),
            "output_mean": self.means[-1],
            "output_range": self.ranges[-1],
        }
        template_global_var = (
                self.template_path / "normalization" /
                "template_normalization_cst_in_global_var_file.c.tpl"
        )

        with Path.open(template_global_var) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def pre_processing(
            self: Self,
            nn_input: np.ndarray,
    ) -> np.ndarray:
        """Compute the output of the pre-processing."""
        inputs = nn_input.flatten()
        for i in range(self.input_size):
            inputs[i] = max(self.mins[i], min(self.maxes[i], inputs[i]))
            inputs[i] = (inputs[i] - self.means[i]) / self.ranges[i]
        return np.reshape(inputs, nn_input.shape)

    def post_processing(
            self: Self,
            nn_output: np.ndarray,
    ) -> np.ndarray:
        """Compute the output of the post-processing."""
        for i in range(len(nn_output)):
            nn_output[i] = nn_output[i] * self.ranges[-1] + self.means[-1]
        return nn_output
