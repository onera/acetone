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
import pystache

class InputLayer(Layers.Layers):

    def __init__(self, idx, size, input_shape):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shape = input_shape
        self.data_format = 'channels_first'
        self.name = 'Input_layer'

    def write_to_function_source_file(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['road'] = self.road

        if(self.data_format == 'channels_last'):
            mustach_hash['channels_last'] = True
            mustach_hash['input_channels'] = self.input_shape[3]
            mustach_hash['input_height'] = self.input_shape[1]
            mustach_hash['input_width'] = self.input_shape[2]
        else:
            mustach_hash['size'] = self.size




        with open('src/templates/layers/template_Input_Layer.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def feedforward(self, input):
        
        return input 