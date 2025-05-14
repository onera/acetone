"""Nearest Resize layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import resize_nearest_factory

from .ResizeNearest import ResizeNearest


# The value in the output tensor are found thanks to a (bi)nearest interpolation
class ResizeNearestDefault(ResizeNearest):
    """ResizeNearest layer with default implementation class."""

    def __init__(self: Self,version:str, **kwargs: int) -> None:
        """Build a ResizeNearest layer with default implementation."""
        super().__init__(**kwargs)
        self.name = "ResizeNearest"
        self.mode = "nearest"
        self.version = version
        self.nearest_mode_mapping = {"round_prefer_floor": self.round_prefer_floor,
                                     "round_prefer_ceil": self.round_prefer_ceil,
                                     "floor": self.floor,
                                     "ceil": self.ceil}

    # Defining the several method to choose the nearest
    def floor(self: Self, x: str, y: str) -> str:
        """Generate floor code."""
        return f"{x} = floor({y});"

    def ceil(self: Self, x: str, y: str) -> str:
        """Generate ceil code."""
        return f"{x} = ceil({y});"

    def round_prefer_floor(self: Self, x: str, y: str) -> str:
        """Generate round prefer floor code."""
        return f"{x} = floor(ceil(2*{y})/2);"

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(
            f"{output_str}[y0 + {self.input_width}*(x0 + {self.input_height}*f)]")

        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width
        mustach_hash["coordinate_transformation_mode_x"] = self.coordinate_transformation_mode_mapping[
            self.coordinate_transformation_mode]("i", 2, "x")
        mustach_hash["coordinate_transformation_mode_y"] = self.coordinate_transformation_mode_mapping[
            self.coordinate_transformation_mode]("j", 3, "y")
        mustach_hash["nearest_mode_x"] = self.nearest_mode_mapping[self.nearest_mode]("x0", "x")
        mustach_hash["nearest_mode_y"] = self.nearest_mode_mapping[self.nearest_mode]("y0", "y")

        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                f"tensor_temp[j + {self.output_width}*(i + {self.output_height}*f)]",
                self.idx,
                f"j + {self.output_width}*(i + {self.output_height}*f)")

        with open(self.template_path / "layers" / "Resize" / "template_ResizeNearest.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

def resize_nearest_default_implementation(
        old_layer: ResizeNearest,
        version: str,
) -> ResizeNearestDefault:
    """Create a ResizeNearest_Default layer using the parameters of old_layer."""
    return ResizeNearestDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        input_shape=[1, old_layer.input_channels, old_layer.input_height, old_layer.input_width],
        activation_function=old_layer.activation_function,
        axes=old_layer.axes,
        coordinate_transformation_mode=old_layer.coordinate_transformation_mode,
        exclude_outside=old_layer.exclude_outside,
        keep_aspect_ratio_policy=old_layer.keep_aspect_ratio_policy,
        boolean_resize=True,
        target_size=old_layer.scale,
        roi=old_layer.roi,
        extrapolation_value=old_layer.extrapolation_value,
        nearest_mode=old_layer.nearest_mode,
        cubic_coeff_a=old_layer.cubic_coeff_a,
    )

resize_nearest_factory.register_implementation(
    None,
    resize_nearest_default_implementation,
)

resize_nearest_factory.register_implementation(
    "default",
    resize_nearest_default_implementation,
)
