"""Resize layer base type definition.

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

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


# The resize Layers
# Only Height and Width are resized
# attribut: a lot of stuff
# input: a tensor to be resized, the desired size or a scale to multiply the size by, the region of interest
# output:the resized tensor
##############  https://onnx.ai/onnx/operators/onnx__Resize.html for more informations
# The strategie is always to go throught the elements, find the coordinate in the original tensor
# and apply a transformation to the value in the original tensor to find the value to enter in the new tensor
class Resize(Layer):
    """Resize layer base class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            input_shape: list,
            activation_function: ActivationFunctions,
            axes: np.ndarray | list = [],
            coordinate_transformation_mode: str = "half_pixel",
            exclude_outside: int = 0,
            keep_aspect_ratio_policy: str = "stretch",
            boolean_resize: None | bool = None,
            target_size: np.ndarray | list = [],
            roi: np.ndarray | list = [],
            extrapolation_value: float = 0.,
            nearest_mode: str = "round_prefer_floor",
            cubic_coeff_a: float = -0.75,
    ) -> None:
        """Instantiate a Resize base layer."""
        super().__init__()
        self.idx = idx
        if type(nearest_mode) is bytes:
            self.nearest_mode = str(nearest_mode)[2:-1]
        else:
            self.nearest_mode = nearest_mode

        self.activation_function = activation_function
        self.cubic_coeff_a = cubic_coeff_a
        self.size = size
        self.axes = axes
        self.exclude_outside = exclude_outside
        self.keep_aspect_ratio_policy = keep_aspect_ratio_policy
        self.roi = roi
        self.name = "Resize"
        self.extrapolation_value = extrapolation_value
        if type(coordinate_transformation_mode) is bytes:
            self.coordinate_transformation_mode = str(coordinate_transformation_mode)[2:-1]
        else:
            self.coordinate_transformation_mode = coordinate_transformation_mode
        self.coordinate_transformation_mode_mapping = {
            "half_pixel": self.half_pixel,
            "half_pixel_symmetric": self.half_pixel_symmetric,
            "pytorch_half_pixel": self.pytorch_half_pixel,
            "align_corners": self.align_corners,
            "asymmetric": self.asymmetric,
            "tf_crop_and_resize": self.tf_crop_and_resize,
        }

        self.coordinate_transformation_mode_implem_mapping = {
            "half_pixel": self.half_pixel_implem,
            "half_pixel_symmetric": self.half_pixel_symmetric_implem,
            "pytorch_half_pixel": self.pytorch_half_pixel_implem,
            "align_corners": self.align_corners_implem,
            "asymmetric": self.asymmetric_implem,
            "tf_crop_and_resize": self.tf_crop_and_resize_implem,
        }

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.scale = [1., 0., 0., 0.]
        if boolean_resize:
            self.scale = target_size
            self.output_channels = int(self.input_channels * target_size[1])  # should be equal to input_channels
            self.output_height = int(self.input_height * target_size[2])
            self.output_width = int(self.input_width * target_size[3])
        else:
            self.output_channels = int(target_size[1])  # should be equal to input_channels
            self.output_height = int(target_size[2])
            self.output_width = int(target_size[3])
            self.scale[1] = self.output_channels / self.input_channels  # should be 1
            self.scale[2] = self.output_height / self.input_height
            self.scale[3] = self.output_width / self.input_width

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Resize (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Resize (size must be int)"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Resize (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Resize (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Resize (must be int)"
            msg += "\n"
        if type(self.coordinate_transformation_mode) is not str:
            msg += "Error: coordinate transformation mode type in Resize (must be str)"
            msg += "\n"
        if type(axes) is not np.ndarray and type(axes) is not list:
            msg += "Error: axes type in Resize (must be numpy array or list)"
            msg += "\n"
        if type(self.exclude_outside) is not int:
            msg += "Error: exclude outside type in Resize (must be int)"
            msg += "\n"
        if type(self.keep_aspect_ratio_policy) is not str:
            msg += "Error: keep aspect ratio policy type in Resize (must be str)"
            msg += "\n"
        if type(self.roi) is not np.ndarray and type(self.roi) is not list:
            msg += "Error: roi type in Resize (must be numpy array or list)"
            msg += "\n"
        if type(self.extrapolation_value) is not float and type(self.extrapolation_value) is not int:
            msg += "Error: extrapolation value type in Resize (must be int or float)"
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += ("Error: size value in Resize "
                    "({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        if self.coordinate_transformation_mode not in ["half_pixel",
                                                       "half_pixel_symmetric",
                                                       "pytorch_half_pixel",
                                                       "align_corners",
                                                       "asymmetric",
                                                       "tf_crop_and_resize"]:
            msg += (f"Error: coordinate transformation mode value in Resize "
                    f"({self.coordinate_transformation_mode})")
            msg += "\n"
        if self.keep_aspect_ratio_policy not in ["stretch",
                                                 "not_larger",
                                                 "not_smaller"]:
            msg += (f"Error: keep aspect ratio policy value in Resize "
                    f"({self.keep_aspect_ratio_policy})")
            msg += "\n"
        if self.exclude_outside not in [0, 1]:
            msg += f"Error: exclude outside value in Resize ({self.exclude_outside})"
            msg += "\n"
        for axe in self.axes:
            if axe < 0 or axe >= 4:
                msg += (f"Error: axe out of bound in Resize "
                        f"({axe} for tensor in 4 dimension with first dimension unused)")
                msg += "\n"
        if self.roi:
            if self.axes:
                if len(self.roi) != 2 * len(self.axes):
                    msg += (f"Error: non consistency between the number of roi and the axes given in Resize"
                            f"({len(self.roi)}!={len(self.axes)})")
                    msg += "\n"
            elif len(self.roi) != 8:
                msg += (f"Error: non consistency between the number of roi given and the size of the tensor in Resize "
                        f"({len(self.roi)}!=8)")
                msg += "\n"
            if any(0 > indice > 1 for indice in self.roi):
                msg += f"Error: roi value is not normalized in Resize ({self.roi})"
                msg += "\n"
        for coeff in self.scale:
            if coeff <= 0:
                msg += f"Error: scale value in Resize ({coeff})"
                msg += "\n"
        if msg:
            raise ValueError(msg)

    @abstractmethod
    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""

    @abstractmethod
    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""

    # Defining the several coordinate transformations. cf documentation
    def half_pixel(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the half-pixel coordinate transformation code."""
        return f"{coord_original} = ({coord_resized} + 0.5)/{self.scale[coord_dim]} - 0.5;"

    def half_pixel_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the half-pixel coordinate transformation code."""
        return (coordinate + 0.5) / self.scale[coordinate_dimension] - 0.5

    def half_pixel_symmetric(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the half-pixel-symmetric coordinate transformation code."""
        if coord_dim == 2:
            target_length = self.output_height * self.scale[2]
            input_length = self.input_height
        else:
            target_length = self.output_width * self.scale[3]
            input_length = self.input_width

        s = f"float adjustment = {int(target_length)}/{target_length};\n"
        s += f"                float center = {input_length}/2;\n"
        s += "                float offset = center*(1 - adjustment);\n"
        s += (f"                {coord_original} = offset + ({coord_resized} + 0.5)"
              f"/{self.scale[coord_dim]} - 0.5;")
        return s

    def half_pixel_symmetric_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the half-pixel-symmetric coordinate transformation code."""
        if coordinate_dimension == 2:
            target_length = self.output_height * self.scale[2]
            input_length = self.input_height
        else:
            target_length = self.output_width * self.scale[3]
            input_length = self.input_width

        adjustment = int(target_length) / target_length
        center = input_length / 2
        offset = center * (1 - adjustment)
        return offset + (coordinate + 0.5) / self.scale[coordinate_dimension] - 0.5

    def pytorch_half_pixel(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the pytorch-half-pixel-symmetric coordinate transformation code."""
        target_length = self.output_height if coord_dim == 2 else self.output_width

        s = f"{coord_original} = "
        if target_length > 1:
            s += f"({coord_resized} + 0.5)/{self.scale[coord_dim]} - 0.5;"
        else:
            s += "0;"
        return s

    def pytorch_half_pixel_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the pytorch-half-pixel-symmetric coordinate transformation code."""
        if coordinate_dimension == 2:
            target_length = self.output_height
        else:
            target_length = self.output_width

        if target_length > 1:
            return (coordinate + 0.5) / self.scale[coordinate_dimension] - 0.5
        return 0

    def align_corners(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the align-corners coordinate transformation code."""
        if coord_dim == 2:
            length_original = self.input_height
            length_resized = self.output_height
        else:
            length_original = self.input_width
            length_resized = self.output_width

        s = (f"{coord_original} = {coord_resized}*({length_original} - 1)"
             f"/({length_resized} - 1);")
        return s

    def align_corners_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the align-corners coordinate transformation code."""
        if coordinate_dimension == 2:
            length_original = self.input_height
            length_resized = self.output_height
        else:
            length_original = self.input_width
            length_resized = self.output_width

        return coordinate * (length_original - 1) / (length_resized - 1)

    def asymmetric(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the asymmetric coordinate transformation code."""
        return f"{coord_original} = {coord_resized}/{self.scale[coord_dim]};"

    def asymmetric_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the asymmetric coordinate transformation code."""
        return coordinate / self.scale[coordinate_dimension]

    def tf_crop_and_resize(
            self: Self,
            coord_resized: str,
            coord_dim: int,
            coord_original: str,
    ) -> str:
        """Generate the tf-crop-and-resize coordinate transformation code."""
        if coord_dim == 2:
            length_original = self.input_height
            length_resized = self.output_height
            start = self.roi[2]
            end = self.roi[6]
        else:
            length_original = self.input_width
            length_resized = self.output_width
            start = self.roi[3]
            end = self.roi[7]

        s = f"{coord_original} = "
        if length_resized > 1:
            s += (f"{start}*({length_original} - 1) + {coord_resized}*({end} - {start})"
                  f"*({length_original} - 1)/({length_resized} - 1);")
        else:
            s += f"0.5*({end} - {start})*({length_original} - 1);\n"
        s += (f"                if({coord_original} < 0) || ({coord_original} > {length_original})"
              f"{coord_original} = {self.extrapolation_value}")
        return s

    def tf_crop_and_resize_implem(
            self: Self,
            coordinate: int,
            coordinate_dimension: int,
    ) -> float:
        """Compute the tf-crop-and-resize coordinate transformation code."""
        if coordinate_dimension == 2:
            length_original = self.input_height
            length_resized = self.output_height
            start = self.roi[2]
            end = self.roi[6]
        else:
            length_original = self.input_width
            length_resized = self.output_width
            start = self.roi[3]
            end = self.roi[7]

        if length_resized > 1:
            coordinate_original = (start * (length_original - 1)
                                   + coordinate * (end - start) * (length_original - 1)
                                   / (length_resized - 1))
        else:
            coordinate_original = 0.5 * (start + end) * (length_original - 1)

        if coordinate_original >= length_original or coordinate_original < 0:
            return self.extrapolation_value
        return coordinate_original
