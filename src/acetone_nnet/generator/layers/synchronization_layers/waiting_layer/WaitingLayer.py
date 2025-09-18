"""Waiting Layer definition.

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
import pystache
from typing_extensions import Any, Self

from acetone_nnet.ir import Layer
from acetone_nnet.versioning.layer_factories import waiting_factory


class WaitingLayer(Layer):
    """Waiting Layer definition."""

    def __init__(
            self:Self,
            original_name:str,
            idx: int,
            src_core: int,
            current_core:int,
            path:int,
    ) -> None:
        """Build a Waiting Layer."""
        super().__init__()
        self.idx = idx
        self.src_core = src_core
        self.current_core = current_core
        self.name = "Waiting_layer"
        self.path = path
        if original_name == "":
            self.original_name = f"{self.name}_{self.idx}"
        else:
            self.original_name = original_name

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.src_core) is not int:
            msg += "Error: dst_core type in Waiting Layer (dst_core must be int)"
            msg += "\n"
        if type(self.current_core) is not int:
            msg += "Error: current_core type in Waiting Layer (current_core must be int)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.src_core == self.current_core:
            msg += (
                f"Error: dst_core and current_core are equals in Waiting Layer "
                f"({self.src_core}=={self.current_core})"
            )
            msg += "\n"
        if msg:
            raise ValueError(msg)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        raise NotImplementedError

    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""


class WaitingLayerDefault(WaitingLayer):
    """Waiting layer with default implementation."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Waiting layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {
            "name": self.name,
            "original_name": self.original_name,
            "idx": f"{self.idx:02d}",
            "src_core": f"{self.src_core:02d}",
            "current_core": f"{self.current_core:02d}",
        }

        with open(
            self.template_path / "parallelization" /
            "synchronization_layers" / "template_Waiting.c.tpl"
        ) as template_file:
            template  = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

def waiting_default_implementation(
        original: WaitingLayer,
        version: str,
) -> WaitingLayerDefault:
    """Create a WaitingLayer_Default layer using the attributes of old_layer."""
    return WaitingLayerDefault(
        version=version,
        original_name=original.original_name,
        idx=original.idx,
        src_core=original.src_core,
        current_core=original.current_core,
        path=original.path,
    )

waiting_factory.register_implementation(
    None,
    waiting_default_implementation,
)
waiting_factory.register_implementation(
    "default",
    waiting_default_implementation,
)

if __name__ == "__main__":
    pass
