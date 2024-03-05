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
import numpy as np
from numpyencoder import NumpyEncoder

def JSON_from_keras_model(keras_model, output_dir_json):
    model_json = keras_model.to_json()
    model_dict =  json.loads(model_json)

    input_layer_size = 1

    for i in range(1, len(keras_model.input.shape)):
        input_layer_size = input_layer_size * keras_model.input.shape[i]

    if keras_model.layers[0].__class__.__name__ == 'InputLayer':
        start = 0

    else:
        start = 1
        model_dict['config']['layers'][0]['config']['size'] = input_layer_size

    i=0
    for layer_json, layer_keras in zip((model_dict['config']['layers'])[start:], keras_model.layers):
        
        layer_size = 1

        if type(layer_keras.output_shape) is list:
            output_shape = layer_keras.output_shape[0]
            input_shape = layer_keras.input_shape[0]
        else:
            output_shape = layer_keras.output_shape
            input_shape = layer_keras.input_shape
        
        for j in range(1, len(output_shape)): # start at idx 1 cause idx 0 represents batch size, so it's None

            layer_size = layer_size * output_shape[j]
        
        layer_json['config']['size'] = layer_size
        layer_json['config']['input_shape'] = input_shape
        layer_json['config']['output_shape'] = output_shape

        if layer_json['class_name'] == 'Dense' or layer_json['class_name'] == 'Conv2D' :
            layer_json['weights'] = np.float32(layer_keras.get_weights()[0])
            layer_json['biases'] = np.float32(layer_keras.get_weights()[1])
            i = i + 2

    with open(output_dir_json, 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=4, cls = NumpyEncoder)
    f.close()


