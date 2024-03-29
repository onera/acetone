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

import code_generator.Layers as Layers
import numpy as np
import pystache
from abc import abstractmethod

class Pooling2D(Layers.Layers):
    def __init__(self, idx, data_format, size, padding, strides, pool_size, input_shape, output_shape, activation_function,**kwargs):
        
        super().__init__()
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = ''
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]

        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]
        
        self.output_channels = self.input_channels

        self.pooling_funtion = ''
        self.local_var = ''
        self.local_var_2 = ''
        self.output_var = ''

        self.activation_function = activation_function

        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.pool_size,self.pool_size, self.strides)

    @abstractmethod    
    def specific_function(self, index, input_of_layer):
        pass

    def write_to_function_source_file(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['road'] = self.road

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(self.output_var)

        mustach_hash['input_channels'] = self.input_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['update_local_vars'] = self.update_local_vars()
        mustach_hash['pool_size'] = self.pool_size
        mustach_hash['strides'] = self.strides
        mustach_hash['pad_left'] = self.pad_left
        mustach_hash['pad_top'] = self.pad_top
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width
        mustach_hash['specific_function'] = self.specific_function('jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)', output_str)

        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str(self.output_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True
        
        with open('src/templates/layers/template_Pooling2D.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def feedforward(self, input):
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        
        output = np.zeros((self.input_channels, self.output_height, self.output_width))
                
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input

        for c in range(self.input_channels):
            for i in range(self.output_height):
                for j in range(self.output_width): 
                    output[c,i,j]= self.pooling_function((input_padded[c, i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]))
        
        return output
