"""LSTM Layer with default implementation type definition.

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

from acetone_nnet.versioning.layer_factories import lstm_factory

from .LSTM import LSTM


class LSTMDefault(LSTM):
    """LSTM layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a LSTM Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

    # The various ways to compute the operation:
        """Generate computation code for nn LSTM."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["original_name"] = self.original_name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["B"] = b
        mustach_hash["activation_function"] = (
            self.activation_function.write_activation_str("output")
        )
        mustach_hash["alpha"] = self.alpha
        mustach_hash["beta"] = self.beta

        with open(
            self.template_path / "layers" / "LSTM" / "template_LSTM_nn.c.tpl"
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["size"] = self.size
        mustach_hash["road"] = self.path

        mustach_hash["patches_size"] = self.output_width * self.output_height
        if self.transpo[0]:
            mustach_hash["LSTM_code"] = self.algo_LSTM_mapping[self.transpo](
                self.output_height,
                self.output_width,
                self.input_height,
                self.previous_layer[0].output_str,
                f"weights_{self.name}_{self.idx:02d}",
            )
        else:
            mustach_hash["LSTM_code"] = self.algo_LSTM_mapping[self.transpo](
                self.output_height,
                self.output_width,
                self.input_width,
                self.previous_layer[0].output_str,
                f"weights_{self.name}_{self.idx:02d}",
            )

        with open(
            self.template_path / "layers" / "LSTM" / "template_LSTM.c.tpl"
        ) as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def LSTM_default_implementation(
    old_layer: LSTM,
    version: str,
) -> LSTMDefault:
    """Create a LSTM_Default layer using the attributes of old_layer."""
    return LSTMDefault(
        version=version,
        original_name=old_layer.original_name,
        idx=old_layer.idx,
        size=old_layer.size,
        weights=old_layer.weights,
        bias=old_layer.biases,
        input_shape=[1, 1, old_layer.input_height, old_layer.input_width],
        output_shape=[1, 1, old_layer.output_height, old_layer.output_width],
        activation_function=old_layer.activation_function,
    )


lstm_factory.register_implementation(
    None,
    LSTM_default_implementation,
)
lstm_factory.register_implementation(
    "default",
    LSTM_default_implementation,
)
