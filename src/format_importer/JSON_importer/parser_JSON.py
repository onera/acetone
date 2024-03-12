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

import json
from itertools import islice
import numpy as np

from src.code_generator.layers import InputLayer, Dense, Conv2D, AveragePooling2D, MaxPooling2D, Softmax
from src.code_generator.activation_functions import Linear, ReLu, Sigmoid, TanH

def create_actv_function_obj(activation_str):

        if activation_str == 'sigmoid':
            return Sigmoid()
        elif activation_str == 'relu':
            return ReLu()
        elif activation_str == 'tanh':
            return TanH()
        elif activation_str == 'linear':
            return Linear()
        elif activation_str == 'softmax':
            return Softmax()

def load_json(file_to_parse):
        
        file = open(file_to_parse, 'r')
        model = json.load(file)

        data_type = model['config']['layers'][0]['config']['dtype']

        if data_type == 'float64':
            data_type = 'double'
            data_type_py = np.float64

        elif data_type == 'float32':
            data_type = 'float'
            data_type_py = np.float32

        elif data_type == 'int':
            data_type = 'long int'
            data_type_py = np.int32

        layers = []

        l_temp = InputLayer(0, model['config']['layers'][0]['config']['size'])

        layers.append(l_temp)

        nb_softmax_layers = 0
        nb_flatten_layers = 0

        for idx, layer in list(islice(enumerate(model['config']['layers']), 1, None)):

            idx += nb_softmax_layers
            idx -= nb_flatten_layers

            if 'activation' in layer['config']:
                if layer['config']['activation'] == 'softmax':
                    layer['config']['activation'] = 'linear'
                    add_softmax_layer = True
                else:
                    add_softmax_layer = False
            else:
                pass
        
            if layer['class_name'] == 'Dense':
                current_layer = Dense(idx, layer['config']['units'], data_type_py(layer['weights']), data_type_py(layer['biases']), create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'Conv2D': 
                current_layer = Conv2D(idx, layer['config']['size'], layer['config']['padding'], layer['config']['strides'][0], layer['config']['kernel_size'][0], layer['config']['dilation_rate'][0], layer['config']['filters'], layer['config']['input_shape'], layer['config']['output_shape'], data_type_py(layer['weights']), data_type_py(layer['biases']), create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'AveragePooling2D':
                current_layer = AveragePooling2D(idx = idx, size = layer['config']['size'], padding = layer['config']['padding'], strides = layer['config']['strides'][0], pool_size = layer['config']['pool_size'][0], input_shape = layer['config']['input_shape'], output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'MaxPooling2D':
                current_layer = MaxPooling2D(idx = idx, size = layer['config']['size'], padding = layer['config']['padding'], strides = layer['config']['strides'][0], pool_size = layer['config']['pool_size'][0], input_shape = layer['config']['input_shape'], output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'Flatten':
                nb_flatten_layers = 1
                continue
            
            l_temp.next_layer.append(current_layer)
            current_layer.previous_layer.append(l_temp)
            l_temp = current_layer
            layers.append(current_layer)

            # Separeted method to generate softmax
            if add_softmax_layer:
                nb_softmax_layers += 1
                current_layer = Softmax(idx+1, l_temp.size)
                l_temp.next_layer.append(current_layer)
                current_layer.previous_layer.append(l_temp)
                l_temp = current_layer
                layers.append(current_layer)

        print("Finished model initialization.")    
        return layers, data_type, data_type_py

def create_actv_function_obj(activation_str):

        if activation_str == 'sigmoid':
            return Sigmoid()
        elif activation_str == 'relu':
            return ReLu()
        elif activation_str == 'tanh':
            return TanH()
        elif activation_str == 'linear':
            return Linear()
        elif activation_str == 'softmax':
            return Softmax()