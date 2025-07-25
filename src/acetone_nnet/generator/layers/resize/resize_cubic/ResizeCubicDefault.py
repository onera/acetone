"""Cubic Resize layer with default implementation type definition.

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
from typing_extensions import Any, Self

from acetone_nnet.versioning.layer_factories import resize_cubic_factory

from .ResizeCubic import ResizeCubic


# The Cubic mode of the Resize layers
# Use a (bi)cubic interpolation to find the new value
class ResizeCubicDefault(ResizeCubic):
    """ResizeCubic layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a ResizeCubic layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["original_name"] = self.original_name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["output_str"] = output_str
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        mustach_hash["activation_function"] = self.activation_function.write_activation_str(
            f"tensor_temp[j + {self.output_width}*(i + {self.output_height}*f)]")

        mustach_hash["cubic_coeff_a"] = self.cubic_coeff_a
        mustach_hash["output_channels"] = self.output_channels
        mustach_hash["output_height"] = self.output_height
        mustach_hash["output_width"] = self.output_width

        if (self.input_height == 1) and (self.input_width > 1):
            mustach_hash["dimension"] = self.input_width
            mustach_hash["coordinate_transformation_mode"] = self.coordinate_transformation_mode_mapping[
                self.coordinate_transformation_mode]("j", 3, "x")
            dimension = "1D"
        elif (self.input_height > 1) and (self.input_width == 1):
            mustach_hash["dimension"] = self.input_height
            mustach_hash["coordinate_transformation_mode"] = self.coordinate_transformation_mode_mapping[
                self.coordinate_transformation_mode]("i", 2, "x")
            dimension = "1D"
        elif (self.input_height > 1) and (self.input_width > 1):
            mustach_hash["input_width"] = self.input_width
            mustach_hash["input_height"] = self.input_height
            mustach_hash["coordinate_transformation_mode_x"] = self.coordinate_transformation_mode_mapping[
                self.coordinate_transformation_mode]("i", 2, "x")
            mustach_hash["coordinate_transformation_mode_y"] = self.coordinate_transformation_mode_mapping[
                self.coordinate_transformation_mode]("j", 3, "y")
            dimension = "2D"

        with open(self.template_dict[dimension]) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def resize_cubic_default_implementation(
        old_layer: ResizeCubic,
        version: str,
) -> ResizeCubicDefault:
    """Create a ResizeCubic_Default layer using the parameters of old_layer."""
    return ResizeCubicDefault(
        version=version,
        original_name=old_layer.original_name,
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

resize_cubic_factory.register_implementation(
    None,
    resize_cubic_default_implementation,
)

resize_cubic_factory.register_implementation(
    "default",
    resize_cubic_default_implementation,
)
