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

import graph.graph_interpretor as graph

from code_generator.layers.Pooling_layers import AveragePooling2D, MaxPooling2D
from code_generator.layers.Conv_layers import Conv2D_6loops, Conv2D_std_gemm, Conv2D_indirect_gemm 
from code_generator.layers.Pad_layers import ConstantPad
from code_generator.layers.Broadcast_layers import Add, Multiply, Subtract, Divide, Maximum, Minimum, Average
from code_generator.layers.Resize_layers import ResizeCubic, ResizeLinear, ResizeNearest
from code_generator.layers import  Concatenate, Input, Dense, Softmax,  Dot, Clip
from code_generator.activation_functions import Linear, ReLu, Sigmoid, TanH

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

def create_conv2d_obj(algorithm, **kwargs):
       
    if '6loops' in algorithm:
        return Conv2D_6loops.Conv2D_6loops(**kwargs)

    elif 'std_gemm' in algorithm:
        return Conv2D_std_gemm.Conv2D_std_gemm(**kwargs)   

    elif 'indirect_gemm' in algorithm:
        return Conv2D_indirect_gemm.Conv2D_indirect_gemm(**kwargs)

def create_resize_obj(mode, **kwargs):

    if mode == 'bicubic':
        return ResizeCubic.ResizeCubic(**kwargs)
    
    elif mode == 'bilinear':
        return ResizeLinear.ResizeLinear(**kwargs)
    
    else:
        return ResizeNearest.ResizeNearest(**kwargs)

def load_json(file_to_parse, conv_algorithm):
        
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

        l_temp = Input.InputLayer(0, model['config']['layers'][0]['config']['size'])

        layers.append(l_temp)

        nb_softmax_layers = 0
        nb_flatten_layers = 0

        for idx, layer in list(islice(enumerate(model['config']['layers']), 1, None)):
            add_softmax_layer = False
            
            idx += nb_softmax_layers
            idx -= nb_flatten_layers

            if 'activation' in layer['config']:
                if layer['config']['activation'] == 'softmax':
                    layer['config']['activation'] = 'linear'
                    add_softmax_layer = True
        
            if layer['class_name'] == 'Dense':
                weights = np.array(data_type_py(layer['weights']))
                if(len(weights.shape) < 4):
                    for i in range(4-len(weights.shape)): 
                        weights = np.expand_dims(weights, axis=-1)
                weights = np.moveaxis(weights, 2, 0)
                current_layer = Dense.Dense(idx=idx,
                                      size=layer['config']['units'],
                                      weights=weights,
                                      biases=data_type_py(layer['biases']),
                                      activation_function=create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'Conv2D': 
                weights = np.array(data_type_py(layer['weights']))
                if(len(weights.shape) < 4):
                    for i in range(4-len(weights.shape)): 
                        weights = np.expand_dims(weights, axis=-1)
                weights = np.moveaxis(weights, 2, 0)
                current_layer = create_conv2d_obj(algorithm = conv_algorithm,
                                                  conv_algorithm = conv_algorithm,
                                                  idx = idx,
                                                  data_format = layer['config']['data_format'],
                                                  size = layer['config']['size'],
                                                  padding = layer['config']['padding'],
                                                  strides = layer['config']['strides'][0],
                                                  kernel_h = layer['config']['kernel_size'][0],
                                                  kernel_w = layer['config']['kernel_size'][1],
                                                  dilation_rate = layer['config']['dilation_rate'][0],
                                                  nb_filters = layer['config']['filters'],
                                                  input_shape = layer['config']['input_shape'],
                                                  output_shape = layer['config']['output_shape'],
                                                  weights = weights, 
                                                  biases = data_type_py(layer['biases']),
                                                  activation_function = create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'AveragePooling2D':
                current_layer = AveragePooling2D.AveragePooling2D(idx = idx,
                                                 data_format = layer['config']['data_format'],
                                                 size = layer['config']['size'],
                                                 padding = layer['config']['padding'],
                                                 strides = layer['config']['strides'][0],
                                                 pool_size = layer['config']['pool_size'][0],
                                                 input_shape = layer['config']['input_shape'],
                                                 output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'MaxPooling2D':
                current_layer = MaxPooling2D.MaxPooling2D(idx = idx,
                                             size = layer['config']['size'],
                                             data_format = layer['config']['data_format'],
                                             padding = layer['config']['padding'],
                                             strides = layer['config']['strides'][0],
                                             pool_size = layer['config']['pool_size'][0],
                                             input_shape = layer['config']['input_shape'],
                                             output_shape = layer['config']['output_shape'],
                                             activation_layer = Linear())
            
            elif layer['class_name'] == 'Flatten':
                nb_flatten_layers = 1
                continue
            
            elif layer['class_name'] == 'Add':
                current_layer = Add.Add(idx = layer['config']['idx'],
                                    size = layer['config']['size'],
                                    input_shapes = layer['config']['input_shape'],
                                    output_shape = layer['config']['output_shape'],
                                    activation_function = Linear())

            elif layer['class_name'] == 'Multiply':
                current_layer = Multiply.Multiply(idx = layer['config']['idx'],
                                         size = layer['config']['size'], 
                                         input_shapes = layer['config']['input_shape'], 
                                         output_shape = layer['config']['output_shape'],
                                         activation_function= Linear())
            
            elif layer['class_name'] == 'Subtract':
                current_layer = Subtract.Subtract(idx = layer['config']['idx'], 
                                         size = layer['config']['size'], 
                                         input_shapes = layer['config']['input_shape'], 
                                         output_shape = layer['config']['output_shape'],
                                         activation_function= Linear())
            
            elif layer['class_name'] == 'Divide':
                current_layer = Divide.Divide(idx = layer['config']['idx'], 
                                       size = layer['config']['size'], 
                                       input_shapes = layer['config']['input_shape'], 
                                       output_shape = layer['config']['output_shape'],
                                       activation_function= Linear())
            
            elif layer['class_name'] == 'Concatenate':
                current_layer = Concatenate.Concatenate(idx = layer['config']['idx'], 
                                            size = layer['config']['size'],
                                            axis = layer['config']['axis'], 
                                            input_shapes = layer['config']['input_shape'], 
                                            output_shape = layer['config']['output_shape'],
                                            activation_function=Linear())
            
            elif layer['class_name'] == 'Maximum':
                current_layer = Maximum.Maximum(idx = layer['config']['idx'],
                                        size = layer['config']['size'],
                                        activation_function= Linear())
                
            elif layer['class_name'] == 'Minimum':
                current_layer = Minimum.Minimum(idx = layer['config']['idx'],
                                        size = layer['config']['size'],
                                        activation_function= Linear())
            
            elif layer['class_name'] == 'Average':
                current_layer = Average.Average(idx = layer['config']['idx'],
                                        size = layer['config']['size'],
                                        activation_function= Linear())
            
            elif layer['class_name'] == 'Dot':
                current_layer = Dot.Dot(idx = layer['config']['idx'],
                                    size = layer['config']['size'],
                                    axis= layer['config']['axes'],
                                    input_shapes = layer['config']['input_shape'], 
                                    output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'Clip':
                current_layer = Clip.Clip(idx = layer['config']['idx'],
                                     size = layer['config']['size'], 
                                     min= layer['config']['min'], 
                                     max=layer['config']['max'])
            
            elif layer['class_name'] == 'UpSampling2D': # Need to make sure that the 'size' attribut of the Layer is renamed
                current_layer = create_resize_obj(mode = layer['config']['interpolation'],
                                                  idx = layer['config']['idx'],
                                                  size = layer['config']['size'],
                                                  input_shape = layer['config']['input_shape'],
                                                  axes = [],
                                                  coordinate_transformation_mode = 'half_pixel',
                                                  exclude_outside = 0,
                                                  keep_aspect_ratio_policy = 'stretch',
                                                  boolean_resize = 1,
                                                  target_size = layer['config']['scale'],
                                                  roi = [],
                                                  extrapolation_value = 0,
                                                  nearest_mode = 'round_prefer_floor',
                                                  activation_function= Linear())
            
            elif layer['class_name'] == 'ZeroPadding2D':
                current_layer = ConstantPad.Constant_Pad(idx = layer['config']['idx'],
                                             size = layer['config']['size'],
                                             pads = layer['config']['padding'],
                                             constant_value = 0,
                                             axes = [],
                                             input_shape = layer['config']['input_shape'],
                                             activation_function = Linear())
            
            elif  'Normalization' in layer['class_name']:
                continue

            elif  'Dropout' in layer['class_name']:
                continue

            elif  layer['class_name'] == 'Reshape':
                continue

            elif  layer['class_name'] == 'Permute':
                continue

            else:
                raise TypeError("Error: layer"+layer['class_name']+" not supported\n")
            
            for i in layer['config']['prev_layer_idx']:
                current_layer.previous_layer.append(layers[i-nb_flatten_layers])
                layers[i-nb_flatten_layers].next_layer.append(current_layer)

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

        layers, listRoad, maxRoad, dict_cst = graph.tri_topo(layers)
        layers = list(map(lambda x:x.find_output_str(dict_cst), layers))
        print("Finished model initialization.")    
        return layers, data_type, data_type_py, listRoad, maxRoad, dict_cst

