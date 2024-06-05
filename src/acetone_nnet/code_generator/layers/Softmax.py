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
import numpy as np
import pystache

class Softmax(Layer):

    def __init__(self, idx:int, size:int, output_shape:list, axis:int|None):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'
        self.axis = axis

        if self.size in (output_shape):
            self.one_dimension = True
        else:
            self.output_channels = output_shape[1]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]
            self.one_dimension = False


        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Softmax (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Softmax (size must be int)")
        if  type(self.axis)!= int and self.axis!=None:
            raise TypeError("Error: axis type in Softmax (axis must be int or None)")
        if not self.one_dimension:
            if type(self.output_channels) != int:
                raise TypeError("Error: output channels type in Softmax (must be int)")
            if type(self.output_height) != int:
                raise TypeError("Error: output height type in Softmax (must be int)")
            if type(self.output_width) != int:
                raise TypeError("Error: output width type in Softmax (must be int)")

        ### Checking value consistency ###
        if not self.one_dimension:
            if self.size != self.output_channels*self.output_height*self.output_width:
                raise ValueError("Error: size value in Softmax ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        if axis not in [1,2,3] and axis != None:
            raise ValueError("Error: axis out of bound in Softmax ("+str(axis)+"for tensor in 4 dimension with first dimension unused)")
        

    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['road'] = self.path
        mustach_hash['output_str'] = output_str

        mustach_hash['1D'] = self.one_dimension

        if self.one_dimension:
            mustach_hash['size'] = self.size
        else:
            mustach_hash['output_channels'] = self.output_channels
            mustach_hash['output_height'] = self.output_height
            mustach_hash['output_width'] = self.output_width

            if self.axis == 1:
                mustach_hash['sum_dimension_1'] = self.output_height
                mustach_hash['sum_dimension_2'] = self.output_width
                mustach_hash['reduced_dimension'] = self.output_channels
                mustach_hash['reduced_position_1'] = 'i + '+str(self.output_width)+'*f'
                mustach_hash['reduced_position_2'] = 'i + '+str(self.output_width)+'*(f + '+str(self.output_height)+'*j)'
                mustach_hash['softmax_indice'] = 'j + '+str(self.output_width)+'*i'

            elif self.axis == 2:
                mustach_hash['sum_dimension_1'] = self.output_channels
                mustach_hash['sum_dimension_2'] = self.output_width
                mustach_hash['reduced_dimension'] = self.output_height
                mustach_hash['reduced_position_1'] = 'i + '+str(self.output_width)+'*f'
                mustach_hash['reduced_position_2'] = 'i + '+str(self.output_width)+'*(j + '+str(self.output_height)+'*f)'
                mustach_hash['softmax_indice'] = 'j + '+str(self.output_width)+'*f'

            elif self.axis == 3:
                mustach_hash['sum_dimension_1'] = self.output_channels
                mustach_hash['sum_dimension_2'] = self.output_height
                mustach_hash['reduced_dimension'] = self.output_width
                mustach_hash['reduced_position_1'] = 'i + '+str(self.output_height)+'*f'
                mustach_hash['reduced_position_2'] = 'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)'
                mustach_hash['softmax_indice'] = 'i + '+str(self.output_height)+'*f'

        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.path)+'[j]', self.idx, 'j')

        with open(self.template_path+'layers/template_Softmax.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def forward_path_layer(self, input:np.ndarray):
        if self.axis:
            input = input.reshape(1,self.output_channels,self.output_height,self.output_width)

        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp, keepdims=1, axis=self.axis)

        return output