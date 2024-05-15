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

from .Resize import Resize
import numpy as np
import math
import pystache

#The mode Linear of the Resize layers
#The value in the output tensor are found thanks to a (bi)linear interpolation
class ResizeLinear(Resize):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = 'linear'
        self.template_dict = {'1D':self.template_path+'layers/Resize/template_ResizeLinear1D.c.tpl',
                              '2D':self.template_path+'layers/Resize/template_ResizeLinear2D.c.tpl'}
        
    def generate_inference_code_layer(self):

        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        if(self.activation_function.name != 'linear'):
            mustach_hash['linear'] = True
            mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)]')

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        
        if ((self.input_height == 1) and (self.input_width > 1)):
            mustach_hash['len_dimension'] = self.input_width-1
            mustach_hash['dimension'] = self.input_width
            mustach_hash['coordinate_transformation_mode'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x')
            dimension = '1D'
            #return self.write_cst_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            mustach_hash['len_dimension'] = self.input_height-1
            mustach_hash['dimension'] = self.input_height
            mustach_hash['coordinate_transformation_mode'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x')
            dimension = '1D'
            #return self.write_cst_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
            mustach_hash['len_height'] = self.input_height-1
            mustach_hash['len_width'] = self.input_width-1
            mustach_hash['input_width'] = self.input_width
            mustach_hash['input_height'] = self.input_height
            mustach_hash['coordinate_transformation_mode_x'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x')
            mustach_hash['coordinate_transformation_mode_y'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y')
            dimension = '2D'
            #return self.write_cst_interpolation_2D()

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('tensor_temp[j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)]',self.idx,'j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)')

        with open(self.template_dict[dimension],'r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template,mustach_hash)

    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        output = np.zeros((self.output_channels,self.output_height,self.output_width))
        for f in range(self.output_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    x = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](i,2)
                    x = np.clip(x,0,self.input_height-1)
                    y = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](j,3)
                    y = np.clip(y,0,self.input_width-1)

                    x0,x1 = math.floor(x), math.floor(x) + 1
                    y0,y1 = math.floor(y), math.floor(y) + 1

                    f11 = input[f,x0,y0]
                    if x1 >= self.input_height:
                        f21 = 0
                    else:
                        f21 = input[f,x1,y0]
                    if x1 >= self.input_height or y1 >= self.input_width:
                        f22 = 0
                    else:
                        f22 = input[f,x1,y1]
                    if y1 >= self.input_width:
                        f12 = 0
                    else:
                        f12 = input[f,x0,y1]
                    
                    output[f,i,j] = (f11*(x1 - x)*(y1 - y) + f21*(x - x0)*(y1 - y) + f12*(x1 - x)*(y - y0) + f22*(x - x0)*(y - y0))/((x1 - x0)*(y1 - y0))
        
        return output
