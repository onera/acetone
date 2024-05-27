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

#extract a list of subtensor from a given tensor
#attribut: axis alongside of which the submatrix will be extracted (if the desired submatrix must have the height, width or channels of the parent tensor)
#input: a tensor
#output: a list of tensor
class Gather(Layer):
    
    def __init__(self, idx:int, size:int, axis:int,  indices:np.ndarray, input_shape:list, output_shape:list, activation_function:ActivationFunctions):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Gather'
        self.indices = indices
        self.axis = axis
        self.output_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Gather (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Gather (size must be int)")
        if type(self.axis) != int:
            raise TypeError("Erro: axis type in Gather (axis must be int)")
        if type(self.indices) != np.ndarray:
            raise TypeError("Error: indices in Gather (indices must be an numpy array)")
        if type(self.output_channels) != int:
            raise TypeError("Error: output channels type in Gather (must be int)")
        if type(self.output_height) != int:
            raise TypeError("Error: output height type in Gather (must be int)")
        if type(self.output_width) != int:
            raise TypeError("Error: output width type in Gather (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in Gather (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in Gather (must be int)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Gather (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in Gather ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        if axis not in [1,2,3]:
            raise ValueError("Error: axis out of bound in Gather ("+str(axis)+"for tensor in 4 dimension with first dimension unused)")
        for indice in self.indices.flatten():
            if indice < 0 or indice >= input_shape[self.axis]:
                raise ValueError("Error: indice out of bound in Gather ("+str(indice)+"out of bound for input of size"+str(input_shape[self.axis])+")")
        
    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[position]')

        mustach_hash['indices_len'] = len(self.indices.flatten())
        mustach_hash['input_width'] = self.input_width
        mustach_hash['input_height'] = self.input_height

        if(self.axis == 1):
            mustach_hash['channels'] = True
            mustach_hash['output_height'] = self.output_height
            mustach_hash['output_width'] = self.output_width
        elif(self.axis == 2):
            mustach_hash['heights'] = True
            mustach_hash['output_channels'] = self.output_channels
            mustach_hash['output_width'] = self.output_width
        elif(self.axis == 3):
            mustach_hash['widths'] = True
            mustach_hash['output_channels'] = self.output_channels
            mustach_hash['output_height'] = self.output_height

        if(self.activation_function.name == 'linear'):
            mustach_hash['linear'] = True

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('tensor_temp[position]',self.idx,'position')

        with open(self.template_path+'layers/template_Gather.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.output_channels,self.input_height,self.input_width)
        return np.take(input, indices=self.indices, axis=self.axis-1)
