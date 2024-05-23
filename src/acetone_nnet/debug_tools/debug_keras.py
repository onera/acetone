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

import keras
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
import numpy as np

def extract_node_outputs(model:Sequential|Functional):
    return [layer.output for layer in model.layers[1:]]

def extract_targets_indices(model:Functional|Sequential,outputs_name:list[str]):
    targets_indices = []
    for name in outputs_name:
        for i in range(len(model.layers)):
            if model.layers[i].output[0] == name:
                targets_indices.append(i)
    return targets_indices

def debug_keras(target_model:Sequential|Functional|str, dataset:np.ndarray, debug_target:list=[], to_save:bool = False, path:str = ''):
    # Loading the model
    if(type(target_model) == str): 
        model = keras.models.load_model(target_model)
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
    functional = [Functional([model.input],[out]) for out in inter_layers]

    # Saving the model
    if to_save:
        for i in range(len(functional)):
            keras.models.save_model(model, path+"model_"+str(i)+".h5")

    # Model inference
    outputs = [func([dataset]) for func in functional]

    for i in range(len(outputs)):
        outputs[i] = outputs[i].ravel().flatten()

    return model, targets_indices, outputs