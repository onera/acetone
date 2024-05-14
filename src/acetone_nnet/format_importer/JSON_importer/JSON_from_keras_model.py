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
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
import json
import numpy as np
from numpyencoder import NumpyEncoder

def JSON_from_keras_model(keras_model:Functional|Sequential, output_dir_json:str):
    model_json = keras_model.to_json()
    model_dict =  json.loads(model_json)

    input_layer_size = 1
    for i in range(1, len(keras_model.input.shape)): #start in idx 1 cause idx 0 represents batch size, so it's None in inference phase
        input_layer_size = input_layer_size * keras_model.input.shape[i]

    model_dict['config']['data_format'] = 'channels_first'
    if (hasattr(layer['config'],'data_format')
        and (layer['config']['data_format'] == 'channels_last') for layer in model_dict['config']['layers']):
        model_dict['config']['data_format'] = 'channels_last'

    if keras_model.layers[0].__class__.__name__ == 'InputLayer':
        start = 0

    else:
        start = 1
        model_dict['config']['layers'][0]['config']['size'] = input_layer_size
        if( model_dict['config']['data_format'] == 'channels_last'):
            if(type(keras_model.input_shape) == list and len(keras_model.input_shape[0]) == 4):
                model_dict['config']['layers'][0]['config']['input_shape'] = [(shape[0], shape[3], shape[1], shape[2]) for shape in keras_model.input_shape]
            elif((type(keras_model.input_shape) is not list) and (len(keras_model.input_shape) == 4)):
                model_dict['config']['layers'][0]['config']['input_shape'] = (keras_model.input_shape[0], keras_model.input_shape[3], keras_model.input_shape[1], keras_model.input_shape[2])
            else:
                model_dict['config']['layers'][0]['config']['input_shape'] = keras_model.input_shape
        else:
            model_dict['config']['layers'][0]['config']['input_shape'] = keras_model.input_shape
        # pretty_json['config']['layers'][0]['config']['dtype'] = "float32"
    i=0
    idx = start
    for layer_json, layer_keras in zip((model_dict['config']['layers'])[start:], keras_model.layers):
        
        layer_size = 1

        if type(layer_keras.output_shape) is list:
            output_shape = layer_keras.output_shape[0]
            input_shape = layer_keras.input_shape[0]
        else:
            output_shape = layer_keras.output_shape
            input_shape = layer_keras.input_shape
        
        if ((model_dict['config']['data_format'] == 'channels_last')):
            if(len(output_shape) == 4):
                output_shape = (output_shape[0], output_shape[3], output_shape[1], output_shape[2])

            if ((type(input_shape) is list) and (len(input_shape[0]) == 4)):
                input_shape = [(shape[0], shape[3], shape[1], shape[2]) for shape in input_shape]
            elif((type(input_shape) is not list) and (len(input_shape) == 4)):
                input_shape = (input_shape[0], input_shape[3], input_shape[1], input_shape[2])

        print(input_shape, output_shape)
        
        for j in range(1, len(output_shape)): #start at idx 1 cause idx 0 represents batch size, so it's None

            layer_size = layer_size * output_shape[j]
        
        if(layer_json['class_name'] == 'UpSampling2D'):
            layer_json['config']['scale'] = layer_json['config']['size']
        
        layer_json['config']['idx'] = idx
        layer_json['config']['size'] = layer_size
        layer_json['config']['input_shape'] = input_shape
        layer_json['config']['output_shape'] = output_shape
        # layer_json['config']['dtype'] = "float32"
        print(layer_json['class_name'], idx)
        layer_json['config']['prev_layer_idx'] = []

        if(('inbound_nodes' in layer_json.keys()) and (layer_json['inbound_nodes'])):
            for prev_layer_json in model_dict['config']['layers'][start:idx]:
                for k in range(len(layer_json['inbound_nodes'][0])):
                    if (prev_layer_json['config']['name'] == layer_json['inbound_nodes'][0][k][0]):
                        layer_json['config']['prev_layer_idx'].append(prev_layer_json['config']['idx']) 
        else:
            layer_json['config']['prev_layer_idx'] = [idx-1]

        if layer_json['class_name'] == 'Dense' or layer_json['class_name'] == 'Conv2D' :
            if layer_json['config']['dtype'] == 'float64':
                weights = np.float64(layer_keras.get_weights()[0])
                if(len(weights.shape) < 3):
                    for i in range(3-len(weights.shape)): 
                        weights = np.expand_dims(weights, axis=-1)
                weights = np.moveaxis(weights, 2, 0)
                if(len(weights.shape) < 4):
                    weights = np.expand_dims(weights, axis=0)
                    
                layer_json['weights'] = weights
                layer_json['biases'] = np.float64(layer_keras.get_weights()[1])

            elif layer_json['config']['dtype'] == 'float32':
                weights = np.float32(layer_keras.get_weights()[0])
                if(len(weights.shape) < 3):
                    for i in range(3-len(weights.shape)): 
                        weights = np.expand_dims(weights, axis=-1)
                weights = np.moveaxis(weights, 2, 0)
                if(len(weights.shape) < 4):
                    weights = np.expand_dims(weights, axis=0)
                
                layer_json['weights'] = weights
                layer_json['biases'] = np.float32(layer_keras.get_weights()[1])

            i = i + 2
        
        if(layer_json['class_name'] == 'BatchNormalization'):
            if layer_json['config']['dtype'] == 'float64':
                layer_json['beta'] = np.float64(layer_keras.get_weights()[1])
                layer_json['gamma'] = np.float64(layer_keras.get_weights()[0])
                layer_json['moving_mean'] = np.float64(layer_keras.get_weights()[2])
                layer_json['moving_var'] = np.float64(layer_keras.get_weights()[3])

            elif layer_json['config']['dtype'] == 'float32':
                layer_json['beta'] = np.float32(layer_keras.get_weights()[1])
                layer_json['gamma'] = np.float32(layer_keras.get_weights()[0])
                layer_json['moving_mean'] = np.float32(layer_keras.get_weights()[2])
                layer_json['moving_var'] = np.float32(layer_keras.get_weights()[3])
        
        idx += 1

    with open(output_dir_json, 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=4, cls = NumpyEncoder)
    f.close()


