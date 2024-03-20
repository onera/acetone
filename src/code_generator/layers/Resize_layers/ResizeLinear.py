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
import numpy as np
import tensorflow as tf

#The mode Linear of the Resize layers
#The value in the output tensor are found thanks to a (bi)linear interpolation
class ResizeLinear(Resize.Resize):
    
    def __init__(self, idx, size, input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size, input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                         scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode = 'linear'
        
    def bilinear_interpolation(self):
        #the equation for the bilinear interpolation
        s = '(f11 * (x2 - x) * (y2 - y) +'
        s+= ' f21 * (x - x1) * (y2 - y) +'
        s+= ' f12 * (x2 - x) * (y - y1) +'
        s+= ' f22 * (x - x1) * (y - y1))' 
        s+= ' / ((x2 - x1) * (y2 - y1));\n'
        return s
    
    def write_cst_interpolation_2D(self):
        #the four points for a 2D interpolation and their values
        s = '    y2 = 0;\n'
        s+= '    y1 = '+str(self.input_height-1)+';\n'
        s+= '    x2 = '+str(self.input_width-1)+';\n'
        s+= '    x1 = 0;\n'
        return s
    
    #a function to write the point used in the interpolation
    def write_function_values_interpolation_2D(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f12 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x2 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f22 = '+output_str+'[y2 + ' + str(self.input_width) + ' * (x2 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f21 = '+output_str+'[y2 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        return s        
    
    #The function to do the interpolation but in 1D
    def write_cst_interpolation_1D_width(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the width of the tensor
        s = '    x2 = '+str(self.input_width-1)+';\n'
        s+= '    x1 = 0;\n'
        return s
    
    def write_function_values_interpolation_1D_width(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x1 + ' + str(self.input_width) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x2 + ' + str(self.input_width) + ' * f];\n'
        return s
    
    def write_cst_interpolation_1D_height(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the height of the tensor
        s = '    x2 = '+str(self.input_height-1)+';\n'
        s+= '    x1 = 0;\n'
        return s

    def write_function_values_interpolation_1D_height(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x1 + ' + str(self.inGemmput_height) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x2 + ' + str(self.input_height) + ' * f];\n'
        return s
    
    def linear_interpolation(self):
        #the equation for the interpolation
        s = '(f11 * (x2 - x) +'
        s+= ' f22 * (x - x1))' 
        s+= ' / (x2 - x1);\n'
        return s
    
    #To differentiate the interpolation needed: prevent division by 0
    def write_cst_interpolation(self):
        if ((self.input_height == 1) and (self.input_width > 1)):
            return self.write_cst_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            return self.write_cst_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
            return self.write_cst_interpolation_2D()
        
    def write_function_values_interpolation(self):
        if ((self.input_height == 1) and (self.input_width > 1)):
            return self.write_function_values_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            return self.write_function_values_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
            return self.write_function_values_interpolation_2D()

    def interpolation(self):
        if ((self.input_height > 1) and (self.input_width >1)):
            return self.bilinear_interpolation()
        else:
            return self.linear_interpolation()
    
    def transforme_coordinate(self,source_file):
        if ((self.input_height == 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x'))
        elif((self.input_height > 1) and (self.input_width == 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))
        elif((self.input_height > 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
    
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write(self.write_cst_interpolation()) #The point used in the interpolation
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write(self.write_function_values_inteGemmrpolation()) #f in the value of the allement f_i_i
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.transforme_coordinate(source_file) #Finding the coordinate in the original tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
        source_file.write(self.interpolation()) #Doing the interpolation to find the output value
        if(self.activation_function.name != 'linear'):
            a = self.activation_function.write_activation_str('output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)]')
            source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+ a)
        source_file.write('            }\n        }\n    }\n\n')

    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bilinear').numpy() #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)
