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

#Concatenate two tensor alongside a given axis
#attribut: axis alongside of which the concatenation will be done
#input: a list of tensor to concatenate
#output: the concatenated tensor 
class Concatenate(Layer):
    def __init__(self, idx:int, size:int, axis:int, input_shapes:list, output_shape:list, activation_function:ActivationFunctions):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = 'Concatenate'
        self.axis = axis
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.idx) == int
        assert type(self.size) == int
        assert all((type(shape) == int for shape in input_shape) for input_shape in self.input_shapes)
        assert self.axis in [1,2,3]
        assert type(self.output_channels) == int
        assert type(self.output_height) == int
        assert type(self.output_width) == int
        assert isinstance(self.activation_function, ActivationFunctions)

        ### Checking value consistency ###
        assert self.size == self.output_channels*self.output_height*self.output_width
        assert self.axis == 1 or all(self.output_channels == input_shape[1] for input_shape in self.input_shapes)
        assert self.axis == 2 or all(self.output_height== input_shape[2] for input_shape in self.input_shapes)
        assert self.axis == 3 or all(self.output_width == input_shape[3] for input_shape in self.input_shapes)
    
    def generate_inference_code_layer(self):
        borne_sup = 0
        borne_inf = 0

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[k]')

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width

        if(self.axis == 1):
            mustach_hash['channels'] = True
        elif(self.axis == 2):
            mustach_hash['heights'] = True
        elif(self.axis == 3):
            mustach_hash['widths'] = True

        mustach_hash['concat'] = []
        for k in range(len(self.previous_layer)):
            borne_sup += self.input_shapes[k][self.axis]

            layer_to_concat = {}
            layer_to_concat['input_width'] = self.input_shapes[k][3]
            layer_to_concat['input_height'] = self.input_shapes[k][2]
            layer_to_concat['output_str'] = self.previous_layer[k].output_str
            layer_to_concat['borne_sup'] = borne_sup
            layer_to_concat['borne_inf'] = borne_inf
            mustach_hash['concat'].append(layer_to_concat)

            borne_inf += self.input_shapes[k][self.axis]


        with open(self.template_path+'layers/template_Concatenate.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def forward_path_layer(self, inputs):
        output = inputs[0]
        output = output.reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        for i in range(1,len(inputs)):
            input = inputs[i]
            input = input.reshape(self.input_shapes[i][1],self.input_shapes[i][2],self.input_shapes[i][3])
            output = np.concatenate((output, input),axis=self.axis - 1) 
        return self.activation_function.compute(output)
