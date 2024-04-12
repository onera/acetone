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

from code_generator.layers.Pooling_layers import AveragePooling2D, MaxPooling2D
from code_generator.layers.Conv_layers import Conv2D_6loops, Conv2D_std_gemm, Conv2D_indirect_gemm 
from code_generator.layers.Pad_layers import EdgePad, WrapPad, ReflectPad, ConstantPad
from code_generator.layers.Broadcast_layers import Add, Multiply, Subtract, Divide, Maximum, Minimum, Average
from code_generator.layers.Resize_layers import ResizeCubic, ResizeLinear, ResizeNearest
from code_generator.layers import  Concatenate, Input, Softmax,  Dot, Gather, Gemm, MatMul, AddBias

import code_generator.activation_functions as activation_functions

###### Utility functions ######

def create_conv2d_obj(algorithm, **kwargs):
       
    if '6loops' in algorithm:
        return Conv2D_6loops.Conv2D_6loops(**kwargs)

    elif 'std_gemm' in algorithm:
        return Conv2D_std_gemm.Conv2D_std_gemm(**kwargs)   

    elif 'indirect_gemm' in algorithm:
        return Conv2D_indirect_gemm.Conv2D_indirect_gemm(**kwargs)
        

def create_resize_obj(mode, **kwargs):
    if mode == b'nearest':
        return ResizeNearest.ResizeNearest(**kwargs)
    
    elif mode == b'linear':
        return ResizeLinear.ResizeLinear(**kwargs)
    
    elif mode == b'cubic':
        return ResizeCubic.ResizeCubic(**kwargs)

def create_pad_obj(mode, **kwargs):
    if mode == b'constant':
        return ConstantPad.Constant_Pad(**kwargs)
    elif mode == b'edge':
        return EdgePad.Edge_pad(**kwargs)
    elif mode == b'wrap':
        return WrapPad.Wrap_pad(**kwargs)
    elif mode == b'reflect':
        return ReflectPad.Reflect_pad(**kwargs)

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
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = 1
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
        
        return Input.InputLayer(idx,size,output_shape,'channels_first')

#Create a layer Softmax
def create_Softmax(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Softmax.Softmax(idx = idx,
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
                                size= size,
                                padding= attributs['auto_pad'],
                                strides= attributs['strides'][0],
                                kernel_h= attributs['kernel_shape'][0], 
                                kernel_w= attributs['kernel_shape'][1], 
                                dilation_rate= attributs['dilations'], 
                                nb_filters= initializers[0].dims[0],
                                input_shape= input_shape, 
                                output_shape= output_shape,
                                weights= np.moveaxis(onnx.numpy_helper.to_array(initializers[0]), 0,3),
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
    return Concatenate.Concatenate(idx,
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
    if('nearest_mode' not in attributs):
        attributs['nearest_mode'] = 'round_prefer_floor'
    if('cubic_coeff_a' not in attributs):
        attributs['cubic_coeff_a'] = -0.75
        
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
                             target_size = targ_size,
                             boolean_resize = bool_resize,
                             roi = roi,
                             extrapolation_value = attributs['extrapolation_value'],
                             nearest_mode = attributs['nearest_mode'],
                             cubic_coeff_a=attributs['cubic_coeff_a'],
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
    return Gather.Gather(idx = idx,
                         size = size,
                         axis = attributs['axis'],
                         indices = onnx.numpy_helper.to_array(initializer),
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
    return Gemm.Gemm(idx = idx,
                       size = size,
                       alpha = attributs['alpha'],
                       beta = attributs['beta'],
                       transA = attributs['transA'],
                       transB = attributs['transB'],
                       weights = onnx.numpy_helper.to_array(B_tensor),
                       bias = onnx.numpy_helper.to_array(C_tensor),
                       input_shape = input_shape,
                       output_shape = output_shape,
                       activation_function = activation_functions.Linear())

def create_MatMul(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    right_tensor = look_for_initializer(node.input[0],model)
    left_tensor = look_for_initializer(node.input[1],model)
    if(left_tensor or right_tensor):
        if(right_tensor and not left_tensor):
            #the weigth is the right tensor:  MatMul(W,T)
            side = True
            weights = onnx.numpy_helper.to_array(right_tensor)
            weights = np.reshape(weights, (get_shape(node.input[1],model)[-1],1,1,output_shape[-1]))
            weights = np.moveaxis(weights, 0,3)
            dict_input[idx] = [node.input[1]]
            input_shape = get_shape(node.input[1],model)
        if(left_tensor and not right_tensor):
            #the weigth is the right tensor:  MatMul(W,T)
            side = False
            weights = onnx.numpy_helper.to_array(left_tensor)
            weights = np.reshape(weights, (1,1,get_shape(node.input[0],model)[-1],output_shape[-1]))
            dict_input[idx] = [node.input[0]]
            input_shape = get_shape(node.input[0],model)
        return MatMul.MatMul(idx = idx,
                             size = size,
                             input_shape = input_shape,
                             weights = weights,
                             side = side,
                             activation_function = activation_functions.Linear())
    else:
        dict_input[idx] = node.input
        input_shapes =[]
        for input in node.input:
            input_shapes.append(get_shape(input,model))
        # to check
        return Dot.Dot(idx = idx,
                       size = size,
                       axis = [-1,-2],
                       input_shapes = input_shapes,
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
    return MaxPooling2D.MaxPooling2D(idx = idx,
                                size = size,
                                padding =attributs['auto_pad'],
                                strides = attributs['strides'][0],
                                pool_size = attributs['kernel_shape'][0],
                                input_shape = input_shape,
                                output_shape = output_shape,
                                activation_function = activation_functions.Linear())

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
    return AveragePooling2D.AveragePooling2D(idx=idx,
                                   size=size, 
                                   padding=attributs['auto_pad'],
                                   strides=attributs['strides'][0], 
                                   pool_size=attributs['kernel_shape'][0], 
                                   input_shape=input_shape,
                                   output_shape=output_shape,
                                   activation_function = activation_functions.Linear())

#Create a layer GlobalAveragePool
def create_GlobalAveragePool(node,idx,dict_input,dict_output,model):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return AveragePooling2D.AveragePooling2D(idx = idx,
                                    size = size,
                                    padding = [0,0,0,0],
                                    strides = 0,
                                    pool_size = input_shape[2],
                                    input_shape = input_shape,
                                    output_shape = output_shape,
                                    activation_function = activation_functions.Linear())

### Broadcats layers ###

#create a layer Add
def create_Add(node,idx,dict_input,dict_output,model):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    right_tensor = look_for_initializer(node.input[0],model)
    left_tensor = look_for_initializer(node.input[1],model)
    if(not right_tensor and not left_tensor):
        input_shapes =[]
        for input in node.input:
            input_shapes.append(get_shape(input,model))
        dict_input[idx] = node.input
        return Add.Add(idx=idx,
                        size=size,
                        input_shapes=input_shapes,
                        output_shape=output_shape,
                        activation_function= activation_functions.Linear())
    else:
        if(right_tensor):
            biases = onnx.numpy_helper.to_array(right_tensor)
            dict_input[idx] = [node.input[1]]
        elif(left_tensor):
            biases = onnx.numpy_helper.to_array(left_tensor)
            dict_input[idx] = [node.input[0]]
        return AddBias.Add_Bias(idx = idx,
                                  size = size,
                                  biases = biases,
                                  activation_function = activation_functions.Linear())
    
#create a layer Div
def create_Div(node,idx,dict_input,dict_output,model):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Divide.Divide(idx=idx,
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
    return Multiply.Multiply(idx=idx,
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
    return Subtract.Subtract(idx=idx,
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
    return Maximum.Maximum(idx=idx,
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
    return Minimum.Minimum(idx=idx,
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
    return Average.Average(idx=idx,
                      size=size,
                      input_shapes=input_shapes,
                      output_shape=output_shape,
                      activation_function= activation_functions.Linear())

###### Dict of all the functions ######
layer_type = {"Softmax":create_Softmax,
         "Conv":create_Conv, 
         "Resize":create_Resize,
         "Pad":create_Pad,
         "Concat":create_Concat,
         "Gather":create_Gather,
         "Gemm":create_Gemm,
         "MatMul":create_MatMul,
         "MaxPool":create_MaxPool,
         "AveragePool":create_AveragePool,
         "GlobalAveragePool":create_GlobalAveragePool,
         "Add":create_Add,
         "Mul":create_Mul,
         "Div":create_Div,
         "Sub":create_Sub,
         "Max":create_Max,
         "Min":create_Min,
         "Mean":create_Avg}


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

#Fuse the activation layer Sigmoide with the prior layer
def fuse_Exp(node,dict_output,model,layers):
    layers[dict_output[node.input[0]]].activation_function = activation_functions.Exponential()
    bypass(node,dict_output,model)

#Fuse the activation layer Sigmoide with the prior layer
def fuse_Log(node,dict_output,model,layers):
    layers[dict_output[node.input[0]]].activation_function = activation_functions.Logarithm()
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
    
    
###### Dict of all the functions ######
activation_layers = {"Relu":fuse_ReLu,
                     "Tanh":fuse_Tanh,
                     "Sigmoid":fuse_Sigmoid,
                     "Clip":fuse_Clip,
                     "Exp":fuse_Exp,
                     "Log":fuse_Log}