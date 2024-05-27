"""
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

import onnx
import numpy as np
import onnxruntime as rt

def clean_inputs(model:onnx.ModelProto):
    while len(model.graph.input) != 1:
        model.graph.input.pop()

def extract_node_outputs(model:onnx.ModelProto):
    activation = ["Relu","Tanh","Sigmoid","Clip","Exp","Log","LeakyRelu"]
    ignored = ["Dropout"]
    ouputs_name = []
    for node in model.graph.node:
        if node.op_type in activation:
            ouputs_name[-1] = node.output[0]
        elif node.op_type in ignored:
            continue
        else:
            ouputs_name.append(node.output[0])
    return ouputs_name

def extract_targets_indices(model:onnx.ModelProto,outputs_name:list[str]):
    targets_indices = []
    for name in outputs_name:
        for i in range(len(model.graph.node)):
            if model.graph.node[i].output[0] == name:
                targets_indices.append(i)
    return targets_indices

def run_model_onnx(model:onnx.ModelProto, dataset:np.ndarray):
    sess = rt.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    onnx_result = sess.run(None,{input_name: dataset})

    onnx_result.append(onnx_result.pop(0))
    
    for i in range(len(onnx_result)):
        onnx_result[i] = onnx_result[i].ravel().flatten()
    
    return onnx_result

def debug_onnx(target_model:onnx.ModelProto|str, dataset:np.ndarray, debug_target:list=[], to_save:bool = False, path:str = '', otpimize_inputs:bool = False):
    # Loading the model
    if type(target_model) == str:
        model = onnx.load(target_model)
    else:
        model = target_model

    # Tensor output name and inidce for acetone debug
    if debug_target == []:
        inter_layers = extract_node_outputs(model)
        targets_indices = []
    else:
        inter_layers = debug_target
        targets_indices = extract_targets_indices(model, inter_layers)

    # Add an output after each name of inter_layers
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in inter_layers:
            value_info_protos.append(node)

    model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.

    # Optimizing the model by removing the useless inputs
    if otpimize_inputs:
        clean_inputs(model)
    
    onnx.checker.check_model(model)

    # Save the model
    if to_save:
        onnx.save(model,path)

    # Model inference
    if type(model) == str:
        run_model = model
    else:
        run_model = model.SerializeToString()
    outputs = run_model_onnx(run_model, dataset)
    
    return model, targets_indices, outputs