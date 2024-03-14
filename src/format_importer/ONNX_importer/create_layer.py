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


import numpy as np
import onnx

import code_generator.layers as layers
import code_generator.activation_functions as activation_functions

###### Utility functions ######

def create_conv2d_obj(algorithm, **kwargs):
       
    if '6loops' in algorithm:
        return layers.Conv2D_6loops(**kwargs)

    elif 'std_gemm' in algorithm:
        return layers.Conv2D_std_gemm(**kwargs)   

    elif 'indirect_gemm' in algorithm:
        return layers.Conv2D_indirect_gemm(**kwargs)
        

def create_resize_obj(mode, **kwargs):
    if mode == b'nearest':
        return layers.ResizeNearest(**kwargs)
    
    elif mode == b'linear':
        return layers.ResizeLinear(**kwargs)
    
    elif mode == b'cubic':
        return layers.ResizeCubic(**kwargs)

def create_pad_obj(mode, **kwargs):
    if mode == b'constant':
        return layers.Constant_Pad(**kwargs)
    elif mode == b'edge':
        return layers.Edge_pad(**kwargs)
    elif mode == b'wrap':
        return layers.Wrap_pad(**kwargs)
    elif mode == b'reflect':
        return layers.Reflect_pad(**kwargs)

#Go find the constant named initialzer_name in model(an onnx model)
def look_for_initializer(initializer_name,model):
    if (initializer_name == ''):
        return []
    for initializer in model.graph.initializer:
        if (initializer.name == initializer_name):
            return initializer
    return []


#Return a dictonnary with {attibut: attribut_values}
def extract_attribut(node):
    attributes = {}
    for attribute in node.attribute:
        attributes[attribute.name] = onnx.helper.get_attribute_value(attribute)
    return attributes

#Find the shape of shape_name in model (an onnx model)
#Depend of the carac of the model. Need value_info in the model
def get_shape(shape_name,model):
    shape = []
    for info in model.graph.value_info:
        if(shape_name == info.name):
            shape = [info.type.tensor_type.shape.dim[i].dim_value for i in range(len(info.type.tensor_type.shape.dim))]
    for input in model.graph.input:
        if(shape_name == input.name):
            shape = [input.type.tensor_type.shape.dim[i].dim_value for i in range(len(input.type.tensor_type.shape.dim))]
    for output in model.graph.output:
        if(shape_name == output.name):
            shape = [output.type.tensor_type.shape.dim[i].dim_value for i in range(len(output.type.tensor_type.shape.dim))]
    if (shape and len(shape)<=3):
        shape = [1 for i in range(3-len(shape))] + shape
    return shape

#Return the size of the layer when given the list output_shape
def find_size(output_shape):
    size = 1
    for i in output_shape:
        if i != 0:
            size*=i
    return size


###### Functions to create a Layer ######

#Create an input layers 
def create_Input_Layer(input_layer,idx,dict_output):
        dict_output[input_layer.name] = idx
        output_shape = [input_layer.type.tensor_type.shape.dim[i].dim_value for i in range(len(input_layer.type.tensor_type.shape.dim))]
        size = find_size(output_shape)
        
        return layers.InputLayer(idx,size)

#Create a layer Softmax
def create_Softmax(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Softmax(idx = idx,
                            size = size)


#Create a layer Conv
def create_Conv(node,idx,dict_input,dict_output,model,conv_algorithm):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = 1
    if ('group' not in attributs):
        attributs['group'] = 1
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'] == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    if ('strides' not in attributs):
        attributs['strides'] = [1,1]
        
    if len(initializers)==2:
        biases = onnx.numpy_helper.to_array(initializers[1])
    else:
        biases = np.zeros(output_shape[1])
        
    return create_conv2d_obj(algorithm=conv_algorithm,
                                conv_algorithm=conv_algorithm,
                                idx= idx,
                                data_format= 'channels_first',
                                size= size,
                                padding= attributs['auto_pad'],
                                strides= attributs['strides'][0],
                                kernel_h= attributs['kernel_shape'][0], 
                                kernel_w= attributs['kernel_shape'][1], 
                                dilation_rate= attributs['dilations'], 
                                nb_filters= initializers[0].dims[0],
                                input_shape= input_shape, 
                                output_shape= output_shape,
                                weights= onnx.numpy_helper.to_array(initializers[0]),
                                biases= biases,
                                activation_function= activation_functions.Linear())
    

#Create a layer Concat
def create_Concat(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    return layers.Concatenate(idx,
                                size, 
                                attributs['axis'],
                                input_shapes,
                                output_shape,
                                activation_function= activation_functions.Linear())
    
#Create a layer Resize
def create_Resize(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if ('axes' not in attributs):
        attributs['axes'] = []
    if('coordinate_transformation_mode' not in attributs):
        attributs['coordinate_transformation_mode'] = 'half_pixel'
    if('exclude_outside' not in attributs):
        attributs['exclude_outside'] = 0
    if('keep_aspect_ratio_policy' not in attributs):
        attributs['keep_aspect_ratio_policy'] = 'stretch'
    if('extrapolation_value' not in attributs):
        attributs['extrapolation_value'] = 0.0
        
    if(initializers[2]):
        #target size
        targ_size = onnx.numpy_helper.to_array(initializers[2])
        bool_resize = 0
    else:
        #scale
        targ_size = onnx.numpy_helper.to_array(initializers[1])
        bool_resize = 1
        
    if(initializers[0]):
        roi = onnx.numpy_helper.to_array(initializers[0])
    else:
        roi = initializers[0]
    
    return create_resize_obj(mode = attributs['mode'],
                             idx = idx,
                             size = size,
                             input_shape = input_shape,
                             axes = attributs['axes'],
                             coordinate_transformation_mode = attributs['coordinate_transformation_mode'],
                             exclude_outside = attributs['exclude_outside'],
                             keep_aspect_ratio_policy = attributs['keep_aspect_ratio_policy'],
                             targ_size = targ_size,
                             boolean_resize = bool_resize,
                             roi = roi,
                             extrapolation_value = attributs['extrapolation_value'],
                             nearest_mode = attributs['nearest_mode'],
                             activation_function = activation_functions.Linear())
    
#create a layer Pad
def create_Pad(node,idx,dict_input,dict_output,model):
    axes = []
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if(len(initializers)==2):
        initializers.append([])
    if(initializers[2]):
        axes = onnx.numpy_helper.to_array(initializers[2])
    return create_pad_obj(mode = attributs['mode'],
                          idx = idx,
                          size = size,
                          pads = onnx.numpy_helper.to_array(initializers[0]),
                          constant_value = onnx.numpy_helper.to_array(initializers[1]),
                          axes = axes,
                          input_shape = input_shape,
                          activation_function = activation_functions.Linear())


#create a layer Gather
def create_Gather(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    initializer = look_for_initializer(node.input[1],model)
    return layers.Gather(idx = idx,
                         size = size,
                         axis = attributs['axis'],
                         indices = initializer,
                         input_shape = input_shape,
                         output_shape = output_shape,
                         activation_function = activation_functions.Linear())

#create a layer Gemm
def create_Gemm(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    B_tensor = look_for_initializer(node.input[1],model)
    if(len(node.input) == 3):
        C_tensor = look_for_initializer(node.input[2],model)
    else:
        C_tensor = 0
    if('transA' not in attributs):
        attributs['transA'] = 0
    if('transB' not in attributs):
        attributs['transB'] = 0
    if('alpha' not in attributs):
        attributs['alpha'] = 1.0
    if('beta' not in attributs):
        attributs['beta'] = 1.0
    return layers.Gemm(idx = idx,
                       size = size,
                       alpha = attributs['alpha'],
                       beta = attributs['beta'],
                       transA = attributs['transA'],
                       transB = attributs['transB'],
                       weight = onnx.numpy_helper.to_array(B_tensor),
                       bias = onnx.numpy_helper.to_array(C_tensor),
                       input_shape = input_shape,
                       output_shape = output_shape,
                       activation_function = activation_functions.Linear())
    

### Pooling layers ###

#Create a layer MaxPool
def create_MaxPool(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = 1
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'] == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    if ('strides' not in attributs):
        attributs['strides'] = [1,1]
    return layers.MaxPooling2D(idx = idx,
                                data_format = 'channels_first',
                                size = size,
                                padding =attributs['auto_pad'],
                                strides = attributs['strides'][0],
                                pool_size = attributs['kernel_shape'][0],
                                input_shape = input_shape,
                                output_shape = output_shape)

#cerate a layer AveragePool
def create_AveragePool(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = 1
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'] == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    if ('strides' not in attributs):
        attributs['strides'] = [1,1]
    return layers.AveragePooling2D(idx=idx,
                                   data_format='channels_first',
                                   size=size, 
                                   padding=attributs['auto_pad'],
                                   strides=attributs['strides'][0], 
                                   pool_size=attributs['kernel_shape'][0], 
                                   input_shape=input_shape,
                                   output_shape=output_shape)

#Create a layer GlobalAveragePool
def create_GlobalAveragePool(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.AveragePooling2D(idx = idx,
                                    data_format = 'channels_first',
                                    size = size,
                                    padding = [0,0,0,0],
                                    strides = 0,
                                    pool_size = input_shape[2],
                                    input_shape = input_shape,
                                    output_shape = output_shape)

### Broadcats layers ###

#create a layer Add
def create_Add(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Add(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

#create a layer Div
def create_Div(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Divide(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

#create a layer Mul
def create_Mul(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Multiply(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

#create a layer Sub
def create_Sub(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Subtract(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())
    
#create a layer Max
def create_Max(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Maximum(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

#create a layer Min
def create_Min(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Minimum(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

#create a layer Average
def create_Avg(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Average(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

### Element Wise layers ###

#create a layer Exp
def create_Exp(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Exponential(idx = idx,
                            size = size)

#create a layer Log
def create_Log(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return layers.Logarithm(idx = idx,
                            size = size)


###### Dict of all the functions ######
layer_type = {"Softmax":create_Softmax,
         "Conv":create_Conv, 
         "Resize":create_Resize,
         "Pad":create_Pad,
         "Concat":create_Concat,
         "Gather":create_Gather,
         "Gemm":create_Gemm,
         "MaxPool":create_MaxPool,
         "AveragePool":create_AveragePool,
         "GlobalAveragePool":create_GlobalAveragePool,
         "Mul":create_Mul,
         "Div":create_Div,
         "Sub":create_Sub,
         "Max":create_Max,
         "Min":create_Min,
         "Mean":create_Avg,
         "Exp":create_Exp,
         "Log":create_Log}


###### Function to deal with the 'non important' layers of the graph ######

#Do the opeartion: Dropout.input = Dropout.output
def bypass(node,dict_output,model):
    dict_output[node.output[0]] = dict_output.pop(node.input[0])

def create_initializer(node,dict_output,model):
    const = model.graph.initializer.add()
    const.name = node.output[0]
    const.data_type = node.attribute[0].t.data_type
    const.raw_data = node.attribute[0].t.raw_data

###### Dict of all the functions ######
unused_layers = {"Dropout":bypass,
                  "Constant":create_initializer,
                  "Unsqueeze":bypass,
                  "Reshape":bypass,
                  "LRN":bypass,
                  "Shape":bypass,
                  "BatchNormalization":bypass}

###### Function to fuse to ONNX layers ######

#Fuse the activation layer ReLu with the prior layer
def fuse_ReLu(node,dict_output,model,layers):
    layers[dict_output[node.input[0]]].activation_function = activation_functions.ReLu()
    bypass(node,dict_output,model)

#Fuse the activation layer Tanh with the prior layer
def fuse_Tanh(node,dict_output,model,layers):
    layers[dict_output[node.input[0]]].activation_function = activation_functions.TanH()
    bypass(node,dict_output,model)

#Fuse the activation layer Sigmoide with the prior layer
def fuse_Sigmoid(node,dict_output,model,layers):
    layers[dict_output[node.input[0]]].activation_function = activation_functions.Sigmoid()
    bypass(node,dict_output,model)

#Fuse a layer Clip with the prior layer
def fuse_Clip(node,dict_output,model,layers):
    min, max = float('-inf'), float('inf')
    if(node.input[1]):
        min = onnx.numpy_helper.to_array(look_for_initializer(node.input[1],model))[0]
    if(node.input[2]):
        max = onnx.numpy_helper.to_array(look_for_initializer(node.input[2],model))[0]
    layers[dict_output[node.input[0]]].activation_function = activation_functions.Clip(max=max,min=min)
    bypass(node,dict_output,model)

def fuse_Add(node,dict_output,model,layers):
    add = activation_functions.Add()
    for input in node.input:
        add.prior_layers.append(layers[dict_output[input]])
        layers[dict_output[input]].activation_function = add
    bypass(node,dict_output,model)
    
    
###### Dict of all the functions ######
activation_layers = {"Relu":fuse_ReLu,
                     "Tanh":fuse_Tanh,
                     "Sigmoid":fuse_Sigmoid,
                     "Clip":fuse_Clip}