"""Base Layer type definition.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing_extensions import Self

from acetone_nnet import templates


@dataclass
class Layer(ABC):
    """Base class for inference layer."""

    def __init__(self: Self) -> None:
        """Build a non-specific layer."""
        self.idx = 0
        self.size = 0
        self.name = ""
        self.next_layer: list[Layer] = []
        self.previous_layer: list[Layer] = []
        self.path: int | None = None
        self.sorted: int | None = None
        self.output_str = ""
        self.fused_layer = None
        self.template_path = (
            str(
                Path(templates.__file__).parent,
            )
            + "/"
        )  # FIXME Should be a proper path

        super().__init__()

    @abstractmethod
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    @abstractmethod
    def forward_path_layer(
        self: Self,
        inputs: np.ndarray | list[np.ndarray],
    ) -> np.ndarray:
        """Compute output of layer."""

    @staticmethod
    def count_elements_array(array: np.ndarray) -> int:
        """Count elements in numpy array."""
        return np.multiply.reduce(array=array.shape)

    @staticmethod
    def compute_padding(
        padding: str | list,
        in_height: int,
        in_width: int,
        kernel_h: int,
        kernel_w: int,
        strides: int,
        dilation_rate: int = 1,
    ) -> tuple[int, int, int, int]:
        """Compute required padding given input and kernel dimensions."""
        pad_right, pad_left, pad_bottom, pad_top = 0, 0, 0, 0
        if isinstance(padding, str):
            # Compute 'same' padding tensorflow

            filter_height = kernel_h - (kernel_h - 1) * (dilation_rate - 1)
            filter_width = kernel_w - (kernel_w - 1) * (dilation_rate - 1)

            # The total padding applied along the height and width is computed as:
            if padding in ("VALID", "valid"):
                pad_right, pad_left, pad_bottom, pad_top = 0, 0, 0, 0
            else:
                if in_height % strides == 0:
                    pad_along_height = max(filter_height - strides, 0)
                else:
                    pad_along_height = max(filter_height - (in_height % strides), 0)
                if in_width % strides == 0:
                    pad_along_width = max(filter_width - strides, 0)
                else:
                    pad_along_width = max(filter_width - (in_width % strides), 0)

                if padding in ("SAME_UPPER", "same"):
                    pad_top = pad_along_height // 2
                    pad_bottom = pad_along_height - pad_top
                    pad_left = pad_along_width // 2
                    pad_right = pad_along_width - pad_left
                elif padding == "SAME_LOWER":
                    pad_bottom = pad_along_height // 2
                    pad_top = pad_along_height - pad_bottom
                    pad_right = pad_along_width // 2
                    pad_left = pad_along_width - pad_right
        else:
            pad_right, pad_left, pad_bottom, pad_top = (
                padding[3],
                padding[1],
                padding[2],
                padding[0],
            )

        return pad_right, pad_left, pad_bottom, pad_top

    # Give to the layer an string saying were the output will be saved (either in a 'cst' or in an 'output_road')
    def find_output_str(self: Self, dict_cst: dict):
        # dict_cst is the dict linking an layer to the cst in which the must be saved if needed

        # either it has to be saved
        if len(dict_cst) and self in dict_cst:
            output_str = "cst_" + str(dict_cst[self])
        # Or it can directly go to the next layer
        else:
            output_str = "output_" + str(self.path)
        self.output_str = output_str
        return self
