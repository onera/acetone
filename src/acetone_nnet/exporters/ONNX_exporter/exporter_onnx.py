"""Exporter to onnx file.

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
import onnx
from onnx import helper

from acetone_nnet.generator.activation_functions import Linear
from acetone_nnet.generator.Layer import Layer
from acetone_nnet.generator.layers import ActivationLayer

from .acetone_to_onnx_layer import (
    export_input,
    export_output,
    list_activation_exporters,
    list_exporters,
)


def onnx_exporter(
        list_layer: list[Layer],
        datatype_py:np.dtype,
        graph_name: str = "ACETONE_graph",
) -> onnx.ModelProto:
    """Export the model to an ONNX model."""
    nodes = []
    initializers = []
    inputs = []
    outputs = []

    for layer in list_layer:
        if not layer.previous_layer:
            inputs.append(export_input(layer, datatype_py))

        if not layer.next_layer:
            outputs.append(export_output(layer, datatype_py))

        exporter = list_exporters.get(layer.name, None)
        if exporter is not None:
            node, initializer = exporter(layer, datatype_py)
            nodes.append(node)
            initializers.extend(initializer)

        if (hasattr(layer, "activation_function")
                and layer.activation_function is not None
                and not isinstance(layer.activation_function, Linear)
                and not isinstance(layer, ActivationLayer)
        ):
            activation_exporter = list_activation_exporters.get(layer.activation_function.name, None)
            if activation_exporter is not None:
                node, initializer = activation_exporter(layer, datatype_py)
                nodes.append(node)
                initializers.extend(initializer)

    graph = helper.make_graph(
        nodes,
        graph_name,
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    return helper.make_model(graph)

