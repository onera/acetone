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

import Resize
import numpy as np
import tensorflow as tf

#The Cubic mode of the Resize layers
#Use a (bi)cubic interpolation to find the new value
class ResizeCubic(Resize.Resize):
    
    def __init__(self, idx, size, input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,
                 cubic_coeff_a = -0.75,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size, input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                         scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode = 'cubic'
        self.cubic_coeff_a = cubic_coeff_a
    
    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bicubic') #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)
    
    #Compute the simple cubic convolution as describe in https://ieeexplore.ieee.org/document/1163711 (cf doc ONNX)
    def cubic_convolution_interpolation(self,f_1,f0,f1,f2,x):
        #give the values to the constants of the interpolation
        s = '                f_1 = ' + f_1 + ';\n'
        s+= '                f0 = ' + f0 + ';\n'
        s+= '                f1 = ' + f1 + ';\n'
        s+= '                f2 = ' + f2 + ';\n'
        s+= '                s = ' + x + '-floor('+x+');\n'
        s+= '                result_interpolation = '
        #the value of the variable of interest: the result of the interpolation
        s+= 'f_1 * a * s * (1 + s * (s - 2)) + '
        s+= 'f0 * (s * s *(a * (s - 1) + 2 * s - 3) + 1) + '
        s+= 'f1 * s * (s * (-s * (2 + a) + 2 * a + 3) - a) + '
        s+= 'f2 * a * s * s * (1 - s);\n'
        return s
    
    #To do the bicubic interpolation, you need to do 4 cubic interpolation alongside a dimension,
    #Then you use this result to do a cubic interpolation alongside the last dimension
    #cf https://en.wikipedia.org/wiki/Bicubic_interpolation
    def bicubic_convolution_interpolation(self,f_1_1,f0_1,f1_1,f2_1,f_10,f00,f10,f20,f_11,f01,f11,f21,f_12,f02,f12,f22):
        s = self.cubic_convolution_interpolation(f_1_1,f0_1,f1_1,f2_1,'x','b_1')
        s+= self.cubic_convolution_interpolation(f_10,f00,f10,f20,'x','b0')
        s+= self.cubic_convolution_interpolation(f_11,f01,f11,f21,'x','b1')
        s+= self.cubic_convolution_interpolation(f_12,f02,f12,f22,'x','b2')
        s+= self.cubic_convolution_interpolation('b_1','b0','b1','b2','y','result_interpolation')
        return s
    
    def transforme_coordinate(self,source_file):
        if ((self.input_height == 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x'))
        elif((self.input_height > 1) and (self.input_width == 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))
        elif((self.input_height > 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
    
    #The function writing the interpoaltion(s) in C
    def interpolation(self):
        output_str = self.previous_layer[0].output_str
        #Setting up the values
        if ((self.input_height > 1) and (self.input_width >1)):
            #All the values used to calculate the output
            f_1_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f0_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f1_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f2_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_10 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f00 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f10 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f20 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_11 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f01 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f11 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f21 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_12 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f02 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f12 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f22 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            return self.bicubic_convolution_interpolation(f_1_1,f0_1,f1_1,f2_1,f_10,f00,f10,f20,f_11,f01,f11,f21,f_12,f02,f12,f22)
        elif((self.input_height > 1) and (self.input_width == 1)):
            f_1 = output_str+'[x0-1 + ' + str(self.input_height) + ' * f];\n'
            f0 = output_str+'[x0 + ' + str(self.input_height) + ' * f];\n'
            f1 = output_str+'[x0+1 + ' + str(self.input_height) + ' * f];\n'
            f2 = output_str+'[x0+2 + ' + str(self.input_height) + ' * f];\n'
            return self.cubic_convolution_interpolation(f_1,f0,f1,f2,'x','result_interpolation')
        elif((self.input_height == 1) and (self.input_width > 1)):
            f_1 = output_str+'[x0-1 + ' + str(self.input_width) + ' * f];\n'
            f0 = output_str+'[x0 + ' + str(self.input_width) + ' * f];\n'
            f1 = output_str+'[x0+1 + ' + str(self.input_width) + ' * f];\n'
            f2 = output_str+'[x0+2 + ' + str(self.input_width) + ' * f];\n'
            return self.cubic_convolution_interpolation(f_1,f0,f1,f2,'x','result_interpolation')
            
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    a = '+str(self.cubic_coeff_a)+';\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.transforme_coordinate(source_file) #Finding the coordinate in the original tensor
        source_file.write('                int x0 = floor(x);\n                int y0 = floor(y);\n')
        source_file.write(self.interpolation())#getting the output
        a = self.activation_function.write_activation_str('result_interpolation')
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        source_file.write('            }\n        }\n    }\n\n')
        
