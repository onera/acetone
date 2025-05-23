"""Debug tools for ONNX.

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

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt


def clean_inputs(model: onnx.ModelProto) -> None:
    """Remove unused inputs from an ONNX model."""
    while len(model.graph.input) != 1:
        model.graph.input.pop()


def extract_node_outputs(model: onnx.ModelProto) -> list[str]:
    """Get outputs name of layers of an ONNX model."""
    activation = ["Relu", "Tanh", "Sigmoid", "Clip", "Exp", "Log", "LeakyRelu"]
    ignored = ["Dropout"]
    outputs_name = []
    for node in model.graph.node:
        if node.op_type in activation:
            outputs_name[-1] = node.output[0]
        elif node.op_type in ignored:
            continue
        else:
            outputs_name.append(node.output[0])
    return outputs_name


def extract_targets_indices(
        model: onnx.ModelProto,
        outputs_name: list[str],
) -> list[int]:
    """Extract the indices each layer will have in ACETONE."""
    targets_indices = []
    for name in outputs_name:
        for i in range(len(model.graph.node)):
            if model.graph.node[i].output[0] == name:
                targets_indices.append(i)
    return targets_indices


def run_model_onnx(
        model: str | Path,
        dataset: np.ndarray,
) -> list[np.ndarray]:
    """Compute the inference of an ONNX model."""
    sess = rt.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    onnx_result = sess.run(None, {input_name: dataset})
    onnx_result.append(onnx_result.pop(0))

    for i in range(len(onnx_result)):
        onnx_result[i] = onnx_result[i].ravel().flatten()

    return onnx_result


def debug_onnx(
        target_model: onnx.ModelProto | str,
        dataset: np.ndarray,
        debug_target: list[str] | None = None,
        to_save: bool = False,
        path: str | Path = "",
        optimize_inputs: bool = False,
) -> (onnx.ModelProto, list[int], list[np.ndarray]):
    """Debug an ONNX model."""
    # Loading the model
    model = onnx.load(target_model) if isinstance(target_model, str) else target_model

    # Tensor output name and indices for acetone debug
    if not debug_target:
        inter_layers = extract_node_outputs(model)
        targets_indices = []
    else:
        inter_layers = debug_target
        targets_indices = extract_targets_indices(model, inter_layers)

    # Add an output after each name of inter_layers
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for _idx, node in enumerate(shape_info.graph.value_info):
        if node.name in inter_layers:
            value_info_protos.append(node)

    model.graph.output.extend(value_info_protos)  # in inference stage, these tensor will be added to output dict.

    # Optimizing the model by removing the useless inputs
    if optimize_inputs:
        clean_inputs(model)

    onnx.checker.check_model(model)

    # Save the model
    if to_save:
        onnx.save(model, path)

    # Model inference
    run_model = model if type(model) is str | Path else model.SerializeToString()
    outputs = run_model_onnx(run_model, dataset)

    return model, targets_indices, outputs
