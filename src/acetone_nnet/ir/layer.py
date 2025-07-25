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

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
from traits.api import (
    ABCHasTraits,
    BaseStr,
    DefaultValue,
    Instance,
    Int,
    List,
    Str,
    Union,
)
from typing_extensions import Self

from acetone_nnet import templates
from acetone_nnet.generator.activation_functions import ActivationFunctions, Linear


class Name(BaseStr):
    """Trait defining a non-empty, string name."""

    info_text = "a non-empty string"

    def get_default_value(self: Self) -> tuple[int, str]:
        """Prevent unspecified name on init."""
        return DefaultValue.disallow, ""

    def validate(self: Self, owner: object, name: str, value: str) -> str | None:
        """Validate Name is a non-empty string."""
        value = super().validate(owner, name, value)  # type: ignore[misc]
        if len(value) > 0:
            return value
        self.error(owner, name, value)
        return None


class Layer(ABCHasTraits):
    """Base class for inference layer."""

    #: The unique index of the layer in its model
    idx = Int(default_value=0)

    # FIXME Should be at worst a computed property,
    #  at best an explicit access to the layer output
    #: The size of the layer output
    size = Int(default_value=0)

    #: Internal layer name (may be used as op sub-type)
    name = Name()

    #: Importer layer name
    original_name = Str()

    #: Preceding layers in the model
    previous_layer = List(Instance("Layer", allow_none=False))

    #: Succeeding layers in the model
    next_layer = List(Instance("Layer", allow_none=False))

    # FIXME Code generation concern, we might need a memory allocator
    #: Identifier for tensor inputs/outputs liveliness
    path = Union(None, Int)

    # FIXME Should be at worst a computed property,
    #  at best an explicit access to the layer output
    #: Allocated output variable name
    output_str = Str()

    #: Fused activation layer, if any
    activation_function = Instance(ActivationFunctions, factory=Linear)

    #: Root path to code generator templates
    template_path: ClassVar[Path] = Path(templates.__file__).parent

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

    # Give to the layer a string saying were the output will be saved
    # (either in a 'cst' or in an 'output_road')
    def find_output_str(self: Self, dict_cst: dict[int, int]) -> Self:
        """Give to the layer a string saying were the output will be saved."""
        # dict_cst is the dict linking a layer to it's cst
        # This cst represent where the output must be saved if needed
        # either it has to be saved
        if len(dict_cst) and self.idx in dict_cst:
            output_str = "cst_" + str(dict_cst[self.idx])
        # Or it can directly go to the next layer
        else:
            output_str = "output_" + str(self.path)
        self.output_str = output_str
        return self

    def __eq__(
        self: Self,
        other: object,
    ) -> bool:
        """Eq method for layers."""
        # compare two layers and say if they are equals
        if type(self) is not type(other):
            return False

        keys = list(self.__dict__.keys())
        for key in keys:
            if (
                key in ("previous_layer", "next_layer", "original_name")
                or type(self.__dict__[key]) is dict
            ):
                continue

            if type(self.__dict__[key]) is np.ndarray:
                if (other.__dict__[key] != self.__dict__[key]).any():
                    return False
            elif other.__dict__[key] != self.__dict__[key]:
                return False
        return True
