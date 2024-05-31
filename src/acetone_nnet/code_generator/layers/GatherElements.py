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
class GatherElements(Layer):
    
    def __init__(self, idx:int, size:int, axis:int,  indices:np.ndarray, input_shape:list, output_shape:list, activation_function:ActivationFunctions):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'GatherElements'
        self.indices = indices
        self.axis = axis
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_channels = output_shape[1]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in GatherElements (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in GatherElements (size must be int)")
        if type(self.axis) != int:
            raise TypeError("Erro: axis type in GatherElements (axis must be int)")
        if type(self.indices) != np.ndarray:
            raise TypeError("Error: indices in GatherElements (indices must be an numpy array)")
        if type(self.output_channels) != int:
            raise TypeError("Error: output channels type in GatherElements (must be int)")
        if type(self.output_height) != int:
            raise TypeError("Error: output height type in GatherElements (must be int)")
        if type(self.output_width) != int:
            raise TypeError("Error: output width type in GatherElements (must be int)")
        if type(self.input_channels) != int:
            raise TypeError("Error: input channels type in GatherElements (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in GatherElements (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in GatherElements (must be int)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in GatherElements (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in GatherElements ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        if axis not in [1,2,3]:
            raise ValueError("Error: axis out of bound in GatherElements ("+str(axis)+"for tensor in 4 dimension with first dimension unused)")
        for indice in self.indices.flatten():
            if indice < 0 or indice >= input_shape[self.axis]:
                raise ValueError("Error: indice out of bound in GatherElements ("+str(indice)+"out of bound for input of size"+str(input_shape[self.axis])+")")
        if self.indices.shape[1:] != (self.output_channels, self.output_height, self.output_width):
            raise ValueError("Error: non consistency between the indice shape and the output shape in GatherElements ("+str(self.indices.shape[1:])+"!="+str((self.output_channels, self.output_height, self.output_width))+")")
        if self.axis != 1 and self.output_channels != self.input_channels:
            raise ValueError("Error: non consistency between the input shape and the output shape in GatherElements ("+str(input_shape)+"!="+str((self.output_channels,self.output_height,self.output_width))+")")
        if self.axis != 2 and self.output_height != self.input_height:
            raise ValueError("Error: non consistency between the input shape and the output shape in GatherElements ("+str(input_shape)+"!="+str((self.output_channels,self.output_height,self.output_width))+")")
        if self.axis != 3 and self.output_width != self.input_width:
            raise ValueError("Error: non consistency between the input shape and the output shape in GatherElements ("+str(input_shape)+"!="+str((self.output_channels,self.output_height,self.output_width))+")")
        
    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[k]')

        mustach_hash['input_width'] = self.input_width
        mustach_hash['input_height'] = self.input_height
        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width

        if(self.axis == 1):
            mustach_hash['channels'] = True
        elif(self.axis == 2):
            mustach_hash['heights'] = True
        elif(self.axis == 3):
            mustach_hash['widths'] = True

        with open(self.template_path+'layers/template_GatherElements.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.input_channels,self.input_height,self.input_width)
        output = np.zeros((self.output_channels, self.output_height, self.output_width))
        for f in range(self.output_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    if self.axis == 1:
                        output[f,i,j] = input[self.indices[0,f,i,j],i,j]
                    elif self.axis == 2:
                        output[f,i,j] = input[f,self.indices[0,f,i,j],j]
                    elif self.axis == 3:
                        output[f,i,j] = input[f,i,self.indices[0,f,i,j]]
        return output
