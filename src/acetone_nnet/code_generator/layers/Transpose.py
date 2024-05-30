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

class Transpose(Layer):

    def __init__(self, idx:int, size:int, input_shape:list, perm:list, activation_function:ActivationFunctions):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Transpose'

        self.perm = perm
    
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.output_channels = input_shape[self.perm[1]]
        self.output_height = input_shape[self.perm[2]]
        self.output_width = input_shape[self.perm[3]]

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Transpose (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Transpose (size must be int)")
        if any(type(indice) != int for indice in self.perm):
            raise TypeError("Error: perm type in Transpose (must be str or ints)")
        if type(self.input_channels) != int:
            raise TypeError("Error: input channels type in Transpose (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in Transpose (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in Transpose (must be int)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Transpose (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in Transpose ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        seen = []
        for indice in self.perm:
            if indice < 0 or indice >= 4:
                raise ValueError("Error: perm out of bound in Transpose ("+str(indice)+"for tensor in 4 dimension with first dimension unused)")
            if indice in seen:
                raise ValueError("Error: non unicity of perm's values in Transpose "+str(self.perm))
            seen.append(indice)
    
    def forward_path_layer(self, input:np.ndarray):
        input = np.reshape(input, (1,self.input_channels, self.input_height, self.input_width))
        return np.transpose(input, self.perm)

    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['size'] = self.size
        mustach_hash['road'] = self.path
        mustach_hash['output_str'] = output_str

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[k]')

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width

        indices = ['Batch','f','i','j']
        if self.perm[1:] == [2,3,1]:
            mustach_hash['a'] = indices[self.perm[1]]
            mustach_hash['b'] = indices[self.perm[3]]
            mustach_hash['c'] = indices[self.perm[2]]
        elif self.perm[1:] == [3,1,2]:
            mustach_hash['a'] = indices[self.perm[2]]
            mustach_hash['b'] = indices[self.perm[1]]
            mustach_hash['c'] = indices[self.perm[3]]
        else:
            mustach_hash['a'] = indices[self.perm[3]]
            mustach_hash['b'] = indices[self.perm[2]]
            mustach_hash['c'] = indices[self.perm[1]]

        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.path)+'[j]', self.idx, 'j')

        with open(self.template_path+'layers/template_Transpose.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)