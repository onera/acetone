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

from .Conv2D import Conv2D

import pystache

class Conv2D_6loops(Conv2D):
    """Implements Conv2D using the six-loops algorithm (direc conv)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(self.local_var)

        mustach_hash['nb_filters'] = self.nb_filters
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['input_channels'] = self.input_channels
        mustach_hash['kernel_h'] = self.kernel_h
        mustach_hash['kernel_w'] = self.kernel_w
        mustach_hash['strides'] = self.strides
        mustach_hash['dilation_rate'] = self.dilation_rate
        mustach_hash['pad_left'] = self.pad_left
        mustach_hash['pad_top'] = self.pad_top
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width

        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str(self.local_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True
        
        with open(self.template_path+'layers/Conv/template_Conv_6loops.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
