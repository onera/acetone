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
from .Pad import Pad

import pystache

#The Edge mode of the Pad layers
#Pads with the edge values of array.
class Edge_pad(Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes, input_shape,activation_function)
        self.mode = 'edge'


    def write_padding(self):
        mustach_hash = {}

        mustach_hash['pads_front'] = self.pads[1]
        mustach_hash['pads_top'] = self.pads[2]
        mustach_hash['pads_left'] = self.pads[3]
        mustach_hash['channels_and_pad_front'] = self.input_shape[1] + self.pads[1]
        mustach_hash['height_and_pad_top'] = self.input_shape[2] + self.pads[2]
        mustach_hash['width_and_pad_left'] = self.input_shape[3] + self.pads[3]

        with open(self.template_path+'layers/Pad/template_Edge_Pad.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    

    def generate_inference_code_layer(self):
        
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['size'] = self.size
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)]')

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['pads_front'] = self.pads[1]
        mustach_hash['pads_top'] = self.pads[2]
        mustach_hash['pads_left'] = self.pads[3]
        mustach_hash['input_width'] = self.input_shape[3]
        mustach_hash['input_height'] = self.input_shape[2]

        mustach_hash['change_indice'] = self.write_padding()

        if(self.activation_function.name == 'linear'):
            mustach_hash['linear'] = True
        
        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('tenser_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)]')

        with open(self.template_path+'layers/Pad/template_Pad_Non_Constant.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)