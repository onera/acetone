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

class Tile(Layer):

    def __init__(self, idx:int, size:int, repeats:list, input_shape:list, activation_function:ActivationFunctions):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.name = 'Tile'
        self.repeats = repeats
        self.output_height = repeats[2]*self.input_height
        self.output_width = repeats[3]*self.input_width
        self.output_channels = repeats[1]*self.input_channels
        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Tile (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Tile (size must be int)")
        if any(type(rep) != int for rep in self.repeats):
            raise TypeError("Error: repeat type in Tile (all must be int")
        if type(self.input_channels) != int:
            raise TypeError("Error: input channels type in Tile (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in Tile (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in Tile (must be int)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Tile (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in Tile ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        if len(self.repeats) != 4:
            raise ValueError("Error: repeats shape in Tile ("+str(len(self.repeats))+" non compatible with input of rank 4)")
        for rep in self.repeats:
            if rep<=0:
                raise ValueError("Error: repeat value in Tile ("+str(rep)+" <= 0)")
    
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape((self.input_channels,self.input_height,self.input_width))
        return np.tile(input, self.repeats[1:])
    
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

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['input_channels'] = self.input_channels
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width

        with open(self.template_path+'layers/template_Tile.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
