"""Reduce layer base type definition.

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
import pystache
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions
from acetone_nnet.generator.Layer import Layer


class Reduce(Layer):
    """Reduce layer base implementation class."""

    def __init__(
            self: Self,
            idx: int,
            size: int,
            axis: tuple[int],
            keepdims: int,
            noop_with_empty_axes: int,
            input_shape: list[int],
            activation_function: ActivationFunctions,
    ) -> None:
        """Instantiate a Reduce layer."""
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = "Reduce"

        self.reduce_func = ""

        self.axes = axis
        self.keepdims = bool(keepdims)
        self.noop_with_empty_axes = noop_with_empty_axes

        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        if 1 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_channels = 1
        else:
            self.output_channels = self.input_channels

        if 2 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_height = 1
        else:
            self.output_height = self.input_height

        if 3 in self.axes or (self.axes == () and not self.noop_with_empty_axes):
            self.output_width = 1
        else:
            self.output_width = self.input_width

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        msg = ""
        if type(self.idx) is not int:
            msg += "Error: idx type in Reduce (idx must be int)"
            msg += "\n"
        if type(self.size) is not int:
            msg += "Error: size type in Reduce (size must be int)"
            msg += "\n"
        if type(self.keepdims) is not bool:
            msg += "Error: keepdims type in Reduce (size must be bool)"
            msg += "\n"
        if type(self.noop_with_empty_axes) is not int:
            msg += "Error: noop with empty axes type in Reduce (size must be int)"
            msg += "\n"
        if any("int" not in type(axe).__name__ for axe in self.axes) or type(self.axes) is not tuple:
            msg += "Error: axes type in Reduce (must be tuple[int])"
            msg += "\n"
        if type(self.input_channels) is not int:
            msg += "Error: input channels type in Reduce (must be int)"
            msg += "\n"
        if type(self.input_height) is not int:
            msg += "Error: input height type in Reduce (must be int)"
            msg += "\n"
        if type(self.input_width) is not int:
            msg += "Error: input width type in Reduce (must be int)"
            msg += "\n"
        if not isinstance(self.activation_function, ActivationFunctions):
            msg += ("Error: activation function type in Reduce "
                    "(activation function must be a sub-classe of acetone_nnet Activation Function)")
            msg += "\n"
        if msg:
            raise TypeError(msg)

        ### Checking value consistency ###
        msg = ""
        if self.size != self.output_channels * self.output_height * self.output_width:
            msg += (f"Error: size value in Reduce "
                    f"({self.size}!={self.output_channels * self.output_height * self.output_width})")
            msg += "\n"
        for axe in self.axes:
            if axe < 0 or axe >= 4:
                msg += (f"Error: axis out of bound in Reduce "
                        f"({axe} for tensor in 4 dimension with first dimension unused)")
                msg += "\n"
        if msg:
            raise ValueError(msg)

    @abstractmethod
    def forward_path_layer(
            self: Self,
            input_array: np.ndarray,
    ) -> np.ndarray:
        """Compute output of layer."""

    def generate_inference_code_layer(self: Self, output_str) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["output_str"] = output_str
        mustach_hash["road"] = self.path
        mustach_hash["size"] = self.size

        which_dim_to_reduc = ""
        if 1 in self.axes:
            which_dim_to_reduc += "c"
        if 2 in self.axes:
            which_dim_to_reduc += "h"
        if 3 in self.axes:
            which_dim_to_reduc += "w"

        if self.reduce_func == "Max":
            mustach_hash["Max"] = True
        elif self.reduce_func == "Min":
            mustach_hash["Min"] = True
        else:
            mustach_hash["Other"] = True
            if self.reduce_func == "Mean":
                mustach_hash["Mean"] = True

        if self.reduce_func in ("Sum", "Mean"):
            mustach_hash["starting_value"] = 0
            mustach_hash["func"] = "+"
        elif self.reduce_func == "Prod":
            mustach_hash["starting_value"] = 1
            mustach_hash["func"] = "*"

        if len(which_dim_to_reduc) == 0:
            if self.noop_with_empty_axes:
                mustach_hash["none"] = True
                if self.activation_function.name != "linear":
                    mustach_hash["Activation"] = True
                    mustach_hash["activation_function"] = self.activation_function.write_activation_str(
                        f"output_{self.path}[k]")
            else:
                mustach_hash["all"] = True
                mustach_hash["activation_function"] = self.activation_function.write_activation_str("reduced")
                mustach_hash["size"] = self.input_channels * self.input_height * self.input_width
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[0]"

        elif len(which_dim_to_reduc) == 1:
            mustach_hash["one"] = True
            mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[k]")

            if which_dim_to_reduc == "c":
                mustach_hash["output_dimension_1"] = self.output_height
                mustach_hash["output_dimension_2"] = self.output_width
                mustach_hash["reduced_dimension"] = self.input_channels
                mustach_hash["position_1"] = f"i + {self.output_width}*f"
                mustach_hash["position_2"] = f"i + {self.input_width}*(f + {self.input_height}*j)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[i + {self.input_width}*f]"

            elif which_dim_to_reduc == "h":
                mustach_hash["output_dimension_1"] = self.output_channels
                mustach_hash["output_dimension_2"] = self.output_width
                mustach_hash["reduced_dimension"] = self.input_height
                mustach_hash["position_1"] = f"i + {self.output_width}*f"
                mustach_hash["position_2"] = f"i + {self.input_width}*(j + {self.input_height}*f)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[i + {self.input_width * self.input_height}*f]"

            elif which_dim_to_reduc == "w":
                mustach_hash["output_dimension_1"] = self.output_channels
                mustach_hash["output_dimension_2"] = self.output_height
                mustach_hash["reduced_dimension"] = self.input_width
                mustach_hash["position_1"] = f"i + {self.output_height}*f"
                mustach_hash["position_2"] = f"j + {self.input_width}*(i + {self.input_height}*f)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[{self.input_width}*(i + {self.input_height}*f)]"

        elif len(which_dim_to_reduc) == 2:
            mustach_hash["two"] = True
            mustach_hash["activation_function"] = self.activation_function.write_activation_str("tensor_temp[k]")

            if which_dim_to_reduc == "ch":
                mustach_hash["output_dimension"] = self.output_width
                mustach_hash["reduced_dimension_1"] = self.input_channels
                mustach_hash["reduced_dimension_2"] = self.input_height
                mustach_hash["position"] = f"f + {self.input_width}*(j + {self.input_height}*i)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[{self.input_width}*f]"

            elif which_dim_to_reduc == "cw":
                mustach_hash["output_dimension"] = self.output_height
                mustach_hash["reduced_dimension_1"] = self.input_channels
                mustach_hash["reduced_dimension_2"] = self.input_width
                mustach_hash["position"] = f"j + {self.input_width}*(f + {self.input_height}*i)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[{self.input_height}*f]"

            elif which_dim_to_reduc == "hw":
                mustach_hash["output_dimension"] = self.output_channels
                mustach_hash["reduced_dimension_1"] = self.input_height
                mustach_hash["reduced_dimension_2"] = self.input_width
                mustach_hash["position"] = f"j + {self.input_width}*(i + {self.input_height}*f)"
                if self.reduce_func in ("Max", "Min"):
                    mustach_hash["starting_value"] = f"{output_str}[{self.input_channels}*f]"

            if self.reduce_func == "Mean":
                mustach_hash["nb_elements"] = mustach_hash["reduced_dimension_1"] * mustach_hash["reduced_dimension_2"]

        elif len(which_dim_to_reduc) == 3:
            mustach_hash["all"] = True
            mustach_hash["activation_function"] = self.activation_function.write_activation_str("reduced")
            mustach_hash["size"] = self.input_channels * self.input_height * self.input_width
            if self.reduce_func in ("Max", "Min"):
                mustach_hash["starting_value"] = output_str + "[0]"

        with open(self.template_path / "layers" / "template_Reduce.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
