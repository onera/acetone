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

from ...Layer import Layer
from ...activation_functions import ActivationFunctions

from abc import abstractmethod
import pystache
import numpy as np

#The class of the Layers which compute operation with broadcast numpy style
#attribut: none
#input: a list of tensor
#output: the resultant tensor
class Broadcast(Layer):
    def __init__(self, idx:int, size:int, input_shapes:list, output_shape:list, activation_function:ActivationFunctions, constant:np.ndarray|float|int|None=None):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ''
        self.input_shapes = input_shapes
        
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.specific_operator = ''
        self.activation_function = activation_function
        self.constant = constant
        if(constant is not None):
            self.constant_size = self.count_elements_array(self.constant)
        
        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.idx) == int
        assert type(self.size) == int
        assert type(self.output_channels) == int
        assert type(self.output_height) == int
        assert type(self.output_width) == int
        assert isinstance(self.activation_function, ActivationFunctions)
        assert type(self.constant) == np.ndarray or type(self.constant) == None

        ### Checking value consistency ###
        assert self.size == self.output_channels*self.output_height*self.output_width
        assert all(shape[1] == 1 or shape[1]==self.output_channels for shape in self.input_shapes)
        assert all(shape[2] == 1 or shape[2]==self.output_height for shape in self.input_shapes)
        assert all(shape[3] == 1 or shape[3]==self.output_width for shape in self.input_shapes)
        assert self.output_channels == np.max(self.input_shapes[:][1])
        assert self.output_height == np.max(self.input_shapes[:][2])
        assert self.output_width == np.max(self.input_shapes[:][3])

    #Go through all the indices and do the operation
    def generate_inference_code_layer(self):

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

        start = 0
        if(self.name == 'Maximum'):
            start = 1
            mustach_hash['output_str0'] = self.previous_layer[0].output_str
            mustach_hash['input_width0'] = self.input_shapes[0][3]
            mustach_hash['input_height0'] = self.input_shapes[0][2]
            mustach_hash['input_channels0'] = self.input_shapes[0][1]
            mustach_hash['max'] = True
        elif(self.name == 'Minimum'):
            start = 1
            mustach_hash['output_str0'] = self.previous_layer[0].output_str
            mustach_hash['input_width0'] = self.input_shapes[0][3]
            mustach_hash['input_height0'] = self.input_shapes[0][2]
            mustach_hash['input_channels0'] = self.input_shapes[0][1]
            mustach_hash['min'] = True
        elif(self.name == 'Average'):
            mustach_hash['Average'] = True
            mustach_hash['prev_size'] = len(self.previous_layer)
        else:
            mustach_hash['other'] = True
        
        mustach_hash['broadcast'] = []
        for k in range(start, len(self.previous_layer)):
            previous_dict = {}
            previous_dict['output_str'] = self.previous_layer[k].output_str
            previous_dict['input_width'] = self.input_shapes[k][3]
            previous_dict['input_height'] = self.input_shapes[k][2]
            previous_dict['input_channels'] = self.input_shapes[k][1]
            if(k != len(self.previous_layer) -1):
                previous_dict['operator'] = self.specific_operator
            mustach_hash['broadcast'].append(previous_dict)
        
        if(self.constant is not None):
            constant_dict = {}
            constant_dict['cst_width'] = self.input_shapes[-1][3]
            constant_dict['cst_height'] = self.input_shapes[-1][2]
            constant_dict['cst_channels'] = self.input_shapes[-1][1]
            constant_dict['operator'] = self.specific_operator
            mustach_hash['constant'] = [constant_dict]

        
        with open(self.template_path+'layers/template_Broadcast.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    @abstractmethod
    def forward_path_layer(self, inputs:np.ndarray):
        pass