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
    ouputs_name = []
    for node in model.graph.node:
        ouputs_name.append(node.output[0])
    return ouputs_name

def debug_onnx(target_model:onnx.ModelProto|str, debug_target:list=[], otpimize_inputs:bool = False, to_save:bool = False, path:str = ''):
    # Loading the model
    if type(target_model) == str:
        model = onnx.load(target_model)
    else:
        model = target_model

    # Tensor output name 
    if debug_target == []:
        inter_layers = extract_node_outputs(model)
    else:
        inter_layers = debug_target

    # Add an output after each name of inter_layers
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in inter_layers:
            print(idx, node)
            value_info_protos.append(node)
    assert len(value_info_protos) == len(inter_layers)
    model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.

    # Optimizing the model by removing the useless inputs
    if otpimize_inputs:
        clean_inputs(model)
    
    onnx.checker.check_model(model)

    # Save the model
    if to_save:
        onnx.save(model,path)
    
    return model

def run_model(model:onnx.ModelProto, dataset:np.ndarray, keep_full_model_result:bool = True):
    sess = rt.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    onnx_result = sess.run(None,{input_name: dataset})

    if not keep_full_model_result:
        onnx_result.pop(0)
    else:
        onnx_result.append(onnx_result.pop(0))
    
    for inter_result in onnx_result:
        inter_result = inter_result.ravel().flatten()
    
    return onnx_result