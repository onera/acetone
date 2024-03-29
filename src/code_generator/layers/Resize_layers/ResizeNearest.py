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

import code_generator.layers.Resize_layers.Resize as Resize
import tensorflow as tf
import numpy as np

#The mode Nearest of the Resize layers.
#The value in the new tensor is found by applying an rounding operation
class ResizeNearest(Resize.Resize):
    
    def __init__(self, idx, size,input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size,input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                        scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode='nearest'
        self.nearest_mode_mapping = {"round_prefer_floor":self.round_prefer_floor,
                                     "round_prefer_ceil":self.round_prefer_ceil,
                                     "floor":self.floor,
                                     "ceil":self.ceil}
    
    #Defining the several method to chose the nearest
    def floor(self,x):
        return '                '+str(x)+' = floor('+str(x)+');\n'
    
    def ceil(self,x,y):
        return '                '+str(x)+' = ceil('+str(y)+');\n'
    
    def round_prefer_floor(self,x,y):
        return '                '+str(x)+' = floor(ceil(2 * ' + str(y) + ') / 2);\n'
    
    def round_prefer_ceil(self,x,y):
        return '                '+str(x)+' = ceil(floor(2 * ' + str(y) + ') / 2);\n'
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
        source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
        source_file.write(self.nearest_mode_mapping[self.nearest_mode]('x'))#Choosing the closest coordinate in the original tensor
        source_file.write(self.nearest_mode_mapping[self.nearest_mode]('y'))
        a = self.activation_function.write_activation_str(output_str+'[ y + ' + str(self.input_width) + ' * (x + ' + str(self.input_height) + ' * f)]')
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        
        source_file.write('            }\n        }\n    }\n\n')
    
    
    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = (tf.image.resize(input, [self.output_height,self.output_width], method='nearest')).numpy() #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)
