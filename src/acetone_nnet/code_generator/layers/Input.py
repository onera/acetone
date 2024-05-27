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
from ..Layer import Layer
import pystache

class InputLayer(Layer):

    def __init__(self, idx:int, size:int, input_shape:list, data_format:str):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shape = input_shape
        if len(self.input_shape) == 4:
            self.output_channels = self.input_shape[1]
            self.output_height = self.input_shape[2]
            self.output_width = self.input_shape[3]
        self.data_format = data_format
        self.name = 'Input_layer'

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Input Layer (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Input Layer (size must be int)")
        if any(type(shape) != int for shape in self.input_shape[1:]):
            raise TypeError("Error: input_shape in Input Layer (all dim must be int)")
        if type(self.data_format) != str:
            raise TypeError("Error: data format type in Input Layer")

        ### Checking value consistency ###
        prod = 1
        for shape in self.input_shape:
            if shape != None and shape != 0:
                prod *= shape

        if self.size != prod:
            raise ValueError("Error: size value in Input Layer ("+str(self.size)+"!="+str(prod)+")")
        if self.data_format not in ['channels_last','channels_first']:
            raise ValueError("Error: data format value in Input Layer ("+self.data_format+")")

    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['road'] = self.path

        if(self.data_format == 'channels_last' and len(self.input_shape) == 4):
            mustach_hash['channels_last'] = True
            mustach_hash['input_channels'] = self.output_channels
            mustach_hash['input_height'] = self.output_height
            mustach_hash['input_width'] = self.output_width
        else:
            mustach_hash['size'] = self.size




        with open(self.template_path+'layers/template_Input_Layer.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def forward_path_layer(self, input:np.ndarray):
        
        return input 