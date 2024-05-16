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
import pystache
import numpy as np

class Flatten(Layer):

    def __init__(self, idx:int, size:int, input_shape:int, data_format:str):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shape = input_shape
        self.data_format = data_format
        self.name = 'Flatten'

        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.idx) == int
        assert type(self.size) == int
        assert all(type(shape) == int for shape in self.input_shape)
        assert self.data_format == 'channels_last' or self.data_format == 'channels_first'

        ### Checking value consistency ###
        assert self.size == self.input_shape[1]*self.input_shape[2]*self.input_shape[3]

    def generate_inference_code_layer(self):

        mustach_hash = {}

        if(self.data_format == 'channels_last'):
            mustach_hash['channels_last'] = True
            mustach_hash['input_channels'] = self.input_shape[1]
            mustach_hash['input_height'] = self.input_shape[2]
            mustach_hash['input_width'] = self.input_shape[3]
            mustach_hash['name'] = self.name
            mustach_hash['idx'] = "{:02d}".format(self.idx)
            mustach_hash['path'] = self.path
            mustach_hash['size'] = self.size

        with open(self.template_path+'layers/template_Flatten.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def forward_path_layer(self, input:np.ndarray):
        if(self.data_format == 'channels_last'):
            return np.transpose(np.reshape(input,self.input_shape[1:]),(1,2,0))
        else:
            return input