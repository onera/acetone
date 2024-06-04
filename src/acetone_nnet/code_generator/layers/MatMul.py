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

from ..Layer import Layer
from ..activation_functions import ActivationFunctions
import numpy as np
import pystache

class MatMul(Layer):

    def __init__(self, idx:int, size:int, input_shapes:list, weights:np.ndarray, side:int, activation_function:ActivationFunctions):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'MatMul'
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        self.side = side
        self.input_shapes = input_shapes

        if weights is not None:
            self.weights = weights
            self.nb_weights = self.count_elements_array(self.weights)
        
        if self.side == 0:
            self.output_channels = self.input_shapes[1]
            self.output_height = self.input_shapes[-2]
            self.output_width = self.weights.shape[-1]
            self.shared_dimension = self.input_shapes[-1]
        elif self.side == 1:
            self.output_channels = self.input_shapes[1]
            self.output_width = self.input_shapes[-1]
            self.output_height = self.weights.shape[-2]
            self.shared_dimension = self.input_shapes[-2]
        elif self.side == 2:
            self.output_channels = self.input_shapes[0][1]
            self.output_height = self.input_shapes[0][-2]
            self.output_width = self.input_shapes[1][-1]
            self.shared_dimension = self.input_shapes[0][-1]

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in MatMul (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in MatMul (size must be int)")
        if type(self.input_shapes[0]) == list:
            for input_shape in self.input_shapes:
                if any(type(shape) != int for shape in input_shape):
                    raise TypeError("Error: input_shape in MatMul (all dim must be int)")
        else:
            if any(type(shape) != int for shape in self.input_shapes):
                raise TypeError("Error: input_shape in MatMul (all dim must be int)")
        if hasattr(self, 'weights'):
            if type(self.weights) != np.ndarray:
                raise TypeError("Error: weights in MatMul (weights must be an numpy array)")
        if type(side) != int:
            raise TypeError("Error: side type in MatMul (side must be a boolean)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in MatMul (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.side == 1:
            if self.weights.shape[-1] != self.input_shapes[-2]:
                raise ValueError("Error: non consistency between weight shape and input shape in MatMul ("+str(self.weights.shape[-1])+"!="+str(self.input_shapes[-2])+")")
            if self.size != self.weights.shape[-2] * self.input_shapes[-1]*self.output_channels:
                raise ValueError("Error: size value in MatMul ("+str(self.size)+" !="+str(self.weights.shape[-2] * self.input_shapes[-1])+")")
        elif self.side == 0:
            if self.weights.shape[-2] != self.input_shapes[-1]:
                raise ValueError("Error: non consistency between weight shape and input shape in MatMul ("+str(self.weights.shape[-2])+"!="+str(self.input_shapes[-1])+")")
            if self.size != self.weights.shape[-1] * self.input_shapes[-2]*self.output_channels:
                raise ValueError("Error: size value in MatMul ("+str(self.size)+" !="+str(self.weights.shape[-1] * self.input_shapes[-2])+")")
        elif self.side == 2:
            if self.input_shapes[1][-2] != self.input_shapes[0][-1]:
                raise ValueError("Error: non consistency between weight shape and input shape in MatMul ("+str(self.input_shapes[1][-2])+"!="+str(self.input_shapes[0][-1])+")")
            if self.size != self.input_shapes[1][-1] * self.input_shapes[0][-2]*self.output_channels:
                raise ValueError("Error: size value in MatMul ("+str(self.size)+" !="+str(self.input_shapes[1][-1] * self.input_shapes[0][-2])+")")
        else: 
            raise ValueError("Error: side value in Matmul (0 is Input*Weight, 1 is Weight*Input, 2 is Input_1*Input_2, "+str(self.side)+" is not implemented)")

    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        if(self.activation_function.name != 'linear'):
            mustach_hash['non_linear'] = True
            mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)]')

        mustach_hash['shared_dimension'] = self.shared_dimension
        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width

        if self.side == 0 :
            mustach_hash['output_str_left'] = self.previous_layer[0].output_str
            mustach_hash['output_str_right'] = 'weights_'+self.name+'_'+"{:02d}".format(self.idx)
        elif self.side == 1:
            mustach_hash['output_str_right'] = self.previous_layer[0].output_str
            mustach_hash['output_str_left'] = 'weights_'+self.name+'_'+"{:02d}".format(self.idx)
        elif self.side == 2:
            mustach_hash['output_str_left'] = self.previous_layer[0].output_str
            mustach_hash['output_str_right'] = self.previous_layer[1].output_str


        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str(self.local_var,self.idx,'i')

        
        
        with open(self.template_path+'layers/template_MatMul.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def forward_path_layer(self, input:np.ndarray|list[np.ndarray]):
        if self.side == 1:
            input_1 = input.reshape(self.input_shapes)
            weights = np.moveaxis(self.weights,3,0)
            weights = np.reshape(weights,(1,1,weights.shape[-1],weights.shape[0]))
            return self.activation_function.compute(np.matmul(weights,input_1))
        elif self.side == 0:
            input_1 = input.reshape(self.input_shapes)
            weights = np.moveaxis(self.weights,3,0)
            weights = np.reshape(weights,(1,1,weights.shape[-1],weights.shape[0]))
            return self.activation_function.compute(np.matmul(input_1,weights))
        elif self.side == 2: 
            input_1 = input[0].reshape(self.input_shapes[0])
            input_2 = input[1].reshape(self.input_shapes[1])
            return self.activation_function.compute(np.matmul(input_1,input_2))

