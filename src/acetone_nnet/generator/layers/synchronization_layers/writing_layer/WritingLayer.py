"""Writing Layer definition.

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
from acetone_nnet.versioning.layer_factories import writing_factory


class WritingLayer(Layer):
    """Writing Layer definition."""

    def __init__(
            self:Self,
            original_name:str,
            idx: int,
            size:int,
            src_core: int,
            current_core:int,
    ) -> None:
        """Build a Writing Layer."""
        super().__init__()
        self.idx = idx
        self.dst_core = src_core
        self.current_core = current_core
        self.name = "WritingLayer"
        if original_name == "":
            self.original_name = f"{self.name}_{self.idx}"
        else:
            self.original_name = original_name
        self.size = size

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.dst_core) is not int:
            msg += "Error: dst_core type in Writing Layer (dst_core must be int)"
            msg += "\n"
        if type(self.current_core) is not int:
            msg += (
                "Error: current_core type in Writing Layer (current_core must be int)"
            )
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.dst_core == self.current_core:
            msg += (
                f"Error: dst_core can't current_core be equals in Writing Layer "
                f"({self.dst_core}=={self.current_core})"
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

class WritingLayerDefault(WritingLayer):
    """Writing layer with default implementation."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Writing layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {
            "name": self.name,
            "original_name": self.original_name,
            "idx": f"{self.idx:02d}",
            "dst_core": f"{self.dst_core:02d}",
            "current_core": f"{self.current_core:02d}",
            "size": f"{self.size:02d}",
        }
        with open(
            self.template_path / "synchronization_layers" / "template_Writing.c.tpl"
        ) as template_file:
            template  = template_file.read()
        template_file.close()
        return pystache.render(template, mustach_hash)

def writing_default_implementation(
        original: WritingLayer,
        version: str,
) -> WritingLayerDefault:
    """Create a WritingLayer_Default layer using the attributes of old_layer."""
    return WritingLayerDefault(
        version=version,
        original_name=original.original_name,
        idx=original.idx,
        src_core=original.dst_core,
        current_core=original.current_core,
        size=original.size,
    )

writing_factory.register_implementation(
    None,
    writing_default_implementation,
)
writing_factory.register_implementation(
    "default",
    writing_default_implementation,
)

if __name__ == "__main__":
    pass
