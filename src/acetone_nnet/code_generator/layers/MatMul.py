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

    def __init__(self, idx:int, size:int, input_shape:list, weights:np.ndarray, side:bool, activation_function:ActivationFunctions):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'MatMul'
        self.weights = weights
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        self.side = side
        self.input_shape = input_shape
        self.nb_weights = self.count_elements_array(self.weights)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in MatMul (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in MatMul (size must be int)")
        if any(type(shape) != int for shape in self.input_shape):
            raise TypeError("Error: input_shape in MatMul (all dim must be int)")
        if type(self.weights) != np.ndarray:
            raise TypeError("Error: weights in MatMul (weights must be an numpy array)")
        if type(side) != bool:
            raise TypeError("Error: side type in MatMul (side must be a boolean)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in MatMul (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.side:
            if self.weights.shape[-1] != self.input_shape[-2]:
                raise ValueError("Error: non consistency between weight shape and input shape in MatMul ("+str(self.weights.shape[-1])+"!="+str(self.input_shape[-2])+")")
            if self.size != self.weights.shape[-2] * self.input_shape[-1]:
                raise ValueError("Error: size value in MatMul ("+str(self.size)+" !="+str(self.weights.shape[-2] * self.input_shape[-1])+")")
        else:
            if self.weights.shape[-2] != self.input_shape[-1]:
                raise ValueError("Error: non consistency between weight shape and input shape in MatMul ("+str(self.weights.shape[-2])+"!="+str(self.input_shape[-1])+")")
            if self.size != self.weights.shape[-1] * self.input_shape[-2]:
                raise ValueError("Error: size value in MatMul ("+str(self.size)+" !="+str(self.weights.shape[-1] * self.input_shape[-2])+")")
        

    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(self.local_var)

        mustach_hash['prev_size'] = self.previous_layer[0].size
        mustach_hash['side'] = self.side

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str(self.local_var,self.idx,'i')

            if(self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True
        
        with open(self.template_path+'layers/template_MatMul.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def forward_path_layer(self, input):
        input = input.reshape(self.input_shape)
        if self.side:
            weights = np.moveaxis(self.weights,3,0)
            weights = np.reshape(weights,(1,1,weights.shape[-1],weights.shape[0]))
            return self.activation_function.compute(np.matmul(weights,input))
        else:
            return self.activation_function.compute(np.matmul(input,self.weights))
