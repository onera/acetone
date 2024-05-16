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

#The mode Nearest of the Resize layers.
#The value in the new tensor is found by applying an rounding operation
class ResizeNearest(Resize):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode='nearest'
        self.nearest_mode_mapping = {"round_prefer_floor":self.round_prefer_floor,
                                     "round_prefer_ceil":self.round_prefer_ceil,
                                     "floor":self.floor,
                                     "ceil":self.ceil}
        
        self.nearest_mode_implem_mapping = {"round_prefer_floor":self.round_prefer_floor_implem,
                                            "round_prefer_ceil":self.round_prefer_ceil_implem,
                                            "floor":math.floor,
                                            "ceil":math.ceil}
        
        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.nearest_mode) == str

        ### Checking value consistency ###
        assert self.nearest_mode in ['round_prefer_floor','round_prefer_ceil','floor','ceil']
    
    #Defining the several method to chose the nearest
    def floor(self, x:str, y:str):
        return x+' = floor('+y+');'
    
    def ceil(self, x:str, y:str):
        return x+' = ceil('+y+');'
    
    def round_prefer_floor(self, x:str, y:str):
        return x+' = floor(ceil(2 * ' + y + ') / 2);'
    
    def round_prefer_floor_implem(self, x:float):
        return math.floor(math.ceil(2*x)/2)
    
    def round_prefer_ceil(self, x:str, y:str):
        return x+' = ceil(floor(2 * ' + y + ') / 2);'
    
    def round_prefer_ceil_implem(self, x:float):
        return math.ceil(math.floor(2*x)/2)
    
    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(output_str+'[y0 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)]')

        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['coordinate_transformation_mode_x'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x')
        mustach_hash['coordinate_transformation_mode_y'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y')
        mustach_hash['nearest_mode_x'] = self.nearest_mode_mapping[self.nearest_mode]('x0','x')
        mustach_hash['nearest_mode_y'] = self.nearest_mode_mapping[self.nearest_mode]('y0','y')

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('tensor_temp[j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)]',self.idx,'j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)')

        with open(self.template_path+'layers/Resize/template_ResizeNearest.c.tpl') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)    
    
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        output = np.zeros((self.output_channels,self.output_height,self.output_width))
        for f in range(self.output_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    x = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](i,2)
                    x0 = self.nearest_mode_implem_mapping[self.nearest_mode](x)
                    y = self.coordinate_transformation_mode_implem_mapping[self.coordinate_transformation_mode](j,3)
                    y0 = self.nearest_mode_implem_mapping[self.nearest_mode](y)

                    output[f,i,j] = input[f,x0,y0]
        return output
