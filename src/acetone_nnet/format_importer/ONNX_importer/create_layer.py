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

from ...code_generator.layers import (
    AveragePooling2D, MaxPooling2D,
    Conv2D_6loops, Conv2D_std_gemm, Conv2D_indirect_gemm,
    Edge_pad, Wrap_pad, Reflect_pad, Constant_Pad,
    Add, Multiply, Subtract, Divide, Maximum, Minimum, Average,
    ResizeCubic, ResizeLinear, ResizeNearest,
    Concatenate, InputLayer, Softmax,  Dot, Gather, Gemm, MatMul, Add_Bias, BatchNormalization
)

from ...code_generator.activation_functions import Linear, ReLu, Sigmoid, TanH, Clip, Exponential, Logarithm, LeakyReLu

###### Utility functions ######

def create_conv2d_obj(algorithm:str, **kwargs):
       
    if '6loops' in algorithm:
        return Conv2D_6loops(**kwargs)

    elif 'std_gemm' in algorithm:
        return Conv2D_std_gemm(**kwargs)   

    elif 'indirect_gemm' in algorithm:
        return Conv2D_indirect_gemm(**kwargs)
        

def create_resize_obj(mode:bytes, **kwargs):
    if mode == b'nearest':
        return ResizeNearest(**kwargs)
    
    elif mode == b'linear':
        return ResizeLinear(**kwargs)
    
    elif mode == b'cubic':
        return ResizeCubic(**kwargs)

def create_pad_obj(mode:bytes, **kwargs):
    if mode == b'constant':
        return Constant_Pad(**kwargs)
    elif mode == b'edge':
        return Edge_pad(**kwargs)
    elif mode == b'wrap':
        return Wrap_pad(**kwargs)
    elif mode == b'reflect':
        return Reflect_pad(**kwargs)

#Go find the constant named initialzer_name in model(an onnx model)
def look_for_initializer(initializer_name:str, model:onnx.ModelProto):
    if (initializer_name == ''):
        return []
    for initializer in model.graph.initializer:
        if (initializer.name == initializer_name):
            return initializer
    return []


#Return a dictonnary with {attibut: attribut_values}
def extract_attribut(node:onnx.NodeProto):
    attributes = {}
    for attribute in node.attribute:
        attributes[attribute.name] = onnx.helper.get_attribute_value(attribute)
    return attributes

#Find the shape of shape_name in model (an onnx model)
#Depend of the carac of the model. Need value_info in the model
def get_shape(shape_name:str, model:onnx.ModelProto):
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
    if (shape and len(shape)<=4):
        shape = [1 for i in range(4-len(shape))] + shape
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = 1
    return shape

#Return the size of the layer when given the list output_shape
def find_size(output_shape:list):
    size = 1
    for i in output_shape:
        if i != 0:
            size*=i
    return size


###### Functions to create a Layer ######

#Create an input layers 
def create_Input_Layer(input_layer:onnx.NodeProto, idx:int, dict_output:dict):
        dict_output[input_layer.name] = idx
        output_shape = [input_layer.type.tensor_type.shape.dim[i].dim_value for i in range(len(input_layer.type.tensor_type.shape.dim))]
        size = find_size(output_shape)
        
        return InputLayer(idx,size,output_shape,'channels_first')

#Create a layer Softmax
def create_Softmax(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Softmax(idx = idx,
                            size = size)


#Create a layer Conv
def create_Conv(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto, conv_algorithm:str):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = [1]
    if ('group' not in attributs):
        attributs['group'] = 1
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'].decode() == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    else:
        attributs['auto_pad'] = attributs['auto_pad'].decode()
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
                                dilation_rate= attributs['dilations'][0], 
                                nb_filters= initializers[0].dims[0],
                                input_shape= input_shape, 
                                output_shape= output_shape,
                                weights= np.moveaxis(onnx.numpy_helper.to_array(initializers[0]), 0,3),
                                biases= biases,
                                activation_function= Linear())
    

#Create a layer Concat
def create_Concat(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    return Concatenate(idx,
                        size, 
                        attributs['axis'],
                        input_shapes,
                        output_shape,
                        activation_function= Linear())
    
#Create a layer Resize
def create_Resize(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if ('axes' not in attributs):
        attributs['axes'] = []
    else:
        for axe in attributs['axes']:
            if axe < 0:
                axe = len(input_shape) - axe
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
        roi = []
    
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
                             activation_function = Linear())
    
#create a layer Pad
def create_Pad(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    axes = []
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    initializers = [look_for_initializer(initializer_name,model) for initializer_name in node.input[1:]]
    attributs = extract_attribut(node)
    if(len(initializers)==2):
        axes = []
    else:
        axes = onnx.numpy_helper.to_array(initializers[2])
        for axe in axes:
            if axe < 0:
                axe = len(input_shape) - axe
    return create_pad_obj(mode = attributs['mode'],
                          idx = idx,
                          size = size,
                          pads = onnx.numpy_helper.to_array(initializers[0]),
                          constant_value = onnx.numpy_helper.to_array(initializers[1]),
                          axes = axes,
                          input_shape = input_shape,
                          activation_function = Linear())


#create a layer Gather
def create_Gather(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    indices = onnx.numpy_helper.to_array(look_for_initializer(node.input[1],model))
    for indice in indices.flatten():
        if indice < 0:
            indice = input_shape[attributs['axis']] - indice
    return Gather(idx = idx,
                    size = size,
                    axis = attributs['axis'],
                    indices = indices,
                    input_shape = input_shape,
                    output_shape = output_shape,
                    activation_function = Linear())

#create a layer Gemm
def create_Gemm(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
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
        C_tensor = np.zeros((1,1))
    if('transA' not in attributs):
        attributs['transA'] = 0
    if('transB' not in attributs):
        attributs['transB'] = 0
    if('alpha' not in attributs):
        attributs['alpha'] = 1.0
    if('beta' not in attributs):
        attributs['beta'] = 1.0
    return Gemm(idx = idx,
                size = size,
                alpha = attributs['alpha'],
                beta = attributs['beta'],
                transA = attributs['transA'],
                transB = attributs['transB'],
                weights = onnx.numpy_helper.to_array(B_tensor),
                bias = onnx.numpy_helper.to_array(C_tensor),
                input_shape = input_shape,
                output_shape = output_shape,
                activation_function = Linear())

def create_MatMul(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
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
            weights = np.reshape(weights, (get_shape(node.input[1],model)[-2],1,1,output_shape[-2]))
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
        return MatMul(idx = idx,
                        size = size,
                        input_shape = input_shape,
                        weights = weights,
                        side = side,
                        activation_function = Linear())
    else:
        dict_input[idx] = node.input
        input_shapes =[]
        for input in node.input:
            input_shapes.append(get_shape(input,model))
        # to check
        return Dot(idx = idx,
                    size = size,
                    axis = [-1,-2],
                    input_shapes = input_shapes,
                    output_shape = output_shape,
                    activation_function = Linear())

def create_BatchNorm(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    
    if('epsilon' not in attributs):
        attributs['epsilon'] = 1e-05
    
    scale = look_for_initializer(node.input[1],model)
    biases = look_for_initializer(node.input[2],model)
    mean = look_for_initializer(node.input[3],model)
    var = look_for_initializer(node.input[4],model)

    return BatchNormalization(idx = idx,
                              size = size,
                              input_shape = output_shape,
                              epsilon = attributs['epsilon'],
                              scale = onnx.numpy_helper.to_array(scale),
                              biases = onnx.numpy_helper.to_array(biases),
                              mean = onnx.numpy_helper.to_array(mean),
                              var = onnx.numpy_helper.to_array(var),
                              activation_function = Linear())



### Pooling layers ###

#Create a layer MaxPool
def create_MaxPool(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = [1]
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'].decode() == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    else:
        attributs['auto_pad'] = attributs['auto_pad'].decode()
    if ('strides' not in attributs):
        attributs['strides'] = [1,1]
    return MaxPooling2D(idx = idx,
                        size = size,
                        padding =attributs['auto_pad'],
                        strides = attributs['strides'][0],
                        pool_size = attributs['kernel_shape'][0],
                        input_shape = input_shape,
                        output_shape = output_shape,
                        activation_function = Linear())

#cerate a layer AveragePool
def create_AveragePool(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = [node.input[0]]
    dict_output[node.output[0]] = idx
    attributs = extract_attribut(node)
    if ('dilations' not in attributs):
        attributs['dilations'] = [1]
    if (('auto_pad' not in attributs) or ( attributs['auto_pad'].decode() == 'NOTSET')):
        attributs['auto_pad'] = attributs['pads']
    else:
        attributs['auto_pad'] = attributs['auto_pad'].decode()
    if ('strides' not in attributs):
        attributs['strides'] = [1,1]
    return AveragePooling2D(idx=idx,
                            size=size, 
                            padding=attributs['auto_pad'],
                            strides=attributs['strides'][0], 
                            pool_size=attributs['kernel_shape'][0], 
                            input_shape=input_shape,
                            output_shape=output_shape,
                            activation_function = Linear())

#Create a layer GlobalAveragePool
def create_GlobalAveragePool(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shape = get_shape(node.input[0],model)
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return AveragePooling2D(idx = idx,
                            size = size,
                            padding = [0,0,0,0],
                            strides = 0,
                            pool_size = input_shape[2],
                            input_shape = input_shape,
                            output_shape = output_shape,
                            activation_function = Linear())

### Broadcats layers ###

#create a layer Add_Biass 
##### UNUSED #####
def create_Add_Biass(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    right_tensor = look_for_initializer(node.input[0],model)
    left_tensor = look_for_initializer(node.input[1],model)
    if(right_tensor):
        biases = onnx.numpy_helper.to_array(right_tensor)
        dict_input[idx] = [node.input[1]]
    elif(left_tensor):
        biases = onnx.numpy_helper.to_array(left_tensor)
        dict_input[idx] = [node.input[0]]
    return Add_Bias(idx = idx,
                        size = size,
                        biases = biases,
                        activation_function = Linear())

#create a layer Add
def create_Add(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    dict_input[idx] = []
    constant = np.zeros(get_shape(node.input[0], model))
    input_shapes =[]
    for input in node.input:
        cst = look_for_initializer(input,model)
        if(cst):
            constant = constant + onnx.numpy_helper.to_array(cst)
        else:
            dict_input[idx].append(input)
            input_shapes.append(get_shape(input,model))
    if(constant.any()):
        if (len(constant.shape)<4):
            for i in range(0,4-len(constant.shape)):
                constant = np.expand_dims(constant,axis=0)
        input_shapes.append(constant.shape)
    else:
        constant = None
    return Add(idx = idx,
                size = size,
                input_shapes = input_shapes,
                output_shape = output_shape,
                activation_function = Linear(),
                constant = constant)
    
#create a layer Div
def create_Div(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    constant = 1
    dict_input[idx] = []
    for input in node.input:
        factor = look_for_initializer(input,model)
        if(factor):
            constant = constant / onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input)
            input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    if((constant == np.ones(constant.shape)).all()):
        constant = None
    else:
        if (len(constant.shape)<4):
            for i in range(0,4-len(constant.shape)):
                constant = np.expand_dims(constant,axis=0)
        input_shapes.append(constant.shape)
    return Divide(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear(),
                    constant=constant)


#create a layer Mul
def create_Mul(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    constant = 1
    dict_input[idx] = []
    for input in node.input:
        factor = look_for_initializer(input,model)
        if(factor):
            constant = constant * onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input)
            input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    if((constant == np.ones(constant.shape)).all()):
        constant = None
    else:
        if (len(constant.shape)<4):
            for i in range(0,4-len(constant.shape)):
                constant = np.expand_dims(constant,axis=0)
        input_shapes.append(constant.shape)
    return Multiply(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear(),
                    constant=constant)

#create a layer Sub
def create_Sub(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    constant = 0
    dict_input[idx] = []
    for input in node.input:
        factor = look_for_initializer(input,model)
        if(factor):
            constant = constant - onnx.numpy_helper.to_array(factor)
        else:
            dict_input[idx].append(input)
            input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_output[node.output[0]] = idx
    if((constant == np.ones(constant.shape)).all()):
        constant = None
    else:
        if (len(constant.shape)<4):
            for i in range(0,4-len(constant.shape)):
                constant = np.expand_dims(constant,axis=0)
        input_shapes.append(constant.shape)
    return Subtract(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear(),
                    constant=constant)
    
#create a layer Max
def create_Max(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Maximum(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear())

#create a layer Min
def create_Min(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Minimum(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear())

#create a layer Average
def create_Avg(node:onnx.NodeProto, idx:int, dict_input:dict, dict_output:dict, model:onnx.ModelProto):
    input_shapes =[]
    for input in node.input:
        input_shapes.append(get_shape(input,model))
    output_shape = get_shape(node.output[0],model)
    size = find_size(output_shape)
    dict_input[idx] = node.input
    dict_output[node.output[0]] = idx
    return Average(idx=idx,
                    size=size,
                    input_shapes=input_shapes,
                    output_shape=output_shape,
                    activation_function= Linear())

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
def bypass(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto):
    dict_output[node.output[0]] = dict_output.pop(node.input[0])

def create_initializer(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto):
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
                  "Flatten":bypass}

###### Function to fuse to ONNX layers ######

#Fuse the activation layer ReLu with the prior layer
def fuse_ReLu(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    layers[dict_output[node.input[0]]].activation_function = ReLu()
    bypass(node,dict_output,model)

#Fuse the activation layer Tanh with the prior layer
def fuse_Tanh(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    layers[dict_output[node.input[0]]].activation_function = TanH()
    bypass(node,dict_output,model)

#Fuse the activation layer Sigmoide with the prior layer
def fuse_Sigmoid(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    layers[dict_output[node.input[0]]].activation_function = Sigmoid()
    bypass(node,dict_output,model)

#Fuse the activation layer Sigmoide with the prior layer
def fuse_Exp(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    layers[dict_output[node.input[0]]].activation_function = Exponential()
    bypass(node,dict_output,model)

#Fuse the activation layer Sigmoid with the prior layer
def fuse_Log(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    layers[dict_output[node.input[0]]].activation_function = Logarithm()
    bypass(node,dict_output,model)

#Fuse a layer Clip with the prior layer
def fuse_Clip(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    min, max = float('-inf'), float('inf')
    if(node.input[1]):
        min = onnx.numpy_helper.to_array(look_for_initializer(node.input[1],model))[0]
    if(node.input[2]):
        max = onnx.numpy_helper.to_array(look_for_initializer(node.input[2],model))[0]
    layers[dict_output[node.input[0]]].activation_function = Clip(max=max,min=min)
    bypass(node,dict_output,model)

#Fuse the activation layer LeakyRelu with the prior layer
def fuse_LeakyReLu(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    attribut = extract_attribut(node)
    if('alpha' not in attribut):
        attribut['alpha'] = 0.01
    layers[dict_output[node.input[0]]].activation_function = LeakyReLu(alpha=attribut['alpha'])
    bypass(node,dict_output,model)

#Fuse a BatchNormalization layer with the previous Conv2D layer
def fuse_BatchNormalization(node:onnx.NodeProto, dict_output:dict, model:onnx.ModelProto, layers:list):
    attributs = extract_attribut(node)
    if('epsilon' not in attributs):
        attributs['epsilon'] = 1e-05
    
    scale = onnx.numpy_helper.to_array(look_for_initializer(node.input[1],model))
    bias = onnx.numpy_helper.to_array(look_for_initializer(node.input[2],model))
    mean = onnx.numpy_helper.to_array(look_for_initializer(node.input[3],model))
    var = onnx.numpy_helper.to_array(look_for_initializer(node.input[4],model))

    weights = layers[dict_output[node.input[0]]].weights
    biases = layers[dict_output[node.input[0]]].biases

    for z in range(len(weights[0,0,0,:])):
        alpha = scale[z]/np.sqrt(var[z] + attributs['epsilon'])
        B = bias[z] - (mean[z]*alpha)
        weights[:,:,:,z] = alpha*weights[:,:,:,z]
        biases[z] = alpha*biases[z] + B
    
    layers[dict_output[node.input[0]]].weights = weights
    layers[dict_output[node.input[0]]].biases = biases

    bypass(node, dict_output, model)

###### Dict of all the functions ######
activation_layers = {"Relu":fuse_ReLu,
                     "Tanh":fuse_Tanh,
                     "Sigmoid":fuse_Sigmoid,
                     "Clip":fuse_Clip,
                     "Exp":fuse_Exp,
                     "Log":fuse_Log,
                     "LeakyRelu":fuse_LeakyReLu}