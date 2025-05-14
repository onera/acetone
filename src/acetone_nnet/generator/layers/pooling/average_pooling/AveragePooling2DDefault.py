"""AveragePooling layer with default implementation type definition.

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

import pystache
from typing_extensions import Self

from acetone_nnet.versioning.layer_factories import average_pooling_factory

from .AveragePooling2D import AveragePooling2D


class AveragePooling2DDefault(AveragePooling2D):
    """AveragePooling layer with default implementation class."""

    def __init__(self: Self, version:str, **kwds: int) -> None:
        """Build an AveragePooling layer."""
        super().__init__(**kwds)
        self.local_var = "sum"
        self.local_var_2 = "count"
        self.output_var = f"{self.local_var}/{self.local_var_2}"
        self.version = version

    def update_local_vars(self: Self) -> str:
        """Generate local var related code."""
        return f"{self.local_var} = 0; {self.local_var_2} = 0;\n"

    def specific_function(self: Self, index: str, input_of_layer: str) -> str:
        """Generate pooling function code."""
        # Computes the average in this subclass AveragePooling2D
        s = f"{self.local_var} += {input_of_layer} [{index}];\n"
        s += f"                            {self.local_var_2}++;\n"

        return s

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(self.output_var)

        mustach_hash["input_channels"] = self.input_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width
        mustach_hash["update_local_vars"] = self.update_local_vars()
        mustach_hash["pool_size"] = self.pool_size
        mustach_hash["strides"] = self.strides
        mustach_hash["pad_left"] = self.pad_left
        mustach_hash["pad_top"] = self.pad_top
        mustach_hash["input_height"] = self.input_height
        mustach_hash["input_width"] = self.input_width
        mustach_hash["specific_function"] = self.specific_function(
            f"jj + {self.input_width}*(ii + {self.input_height}*f)", output_str)

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                self.output_var,
                self.idx,
                f"j + {self.output_width}*(i + {self.output_height}*f)")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "template_Pooling2D.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

def average_pooling_default_implementation(
        old_layer: AveragePooling2D,
        version:str,
) -> AveragePooling2DDefault:
    """Create an AveragePooling2D_Default layer using the attributes of old_layer."""
    return AveragePooling2DDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        padding=old_layer.padding,
        strides=old_layer.strides,
        pool_size=old_layer.pool_size,
        input_shape=[1,old_layer.input_channels,old_layer.input_height,old_layer.input_width],
        output_shape=[1,old_layer.output_channels,old_layer.output_height,old_layer.output_width],
        activation_function=old_layer.activation_function,
    )

average_pooling_factory.register_implementation(
    None,
    average_pooling_default_implementation,
)
average_pooling_factory.register_implementation(
    "default",
    average_pooling_default_implementation,
)
