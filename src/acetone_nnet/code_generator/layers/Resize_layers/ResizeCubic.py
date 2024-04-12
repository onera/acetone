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
import pystache

#The Cubic mode of the Resize layers
#Use a (bi)cubic interpolation to find the new value
class ResizeCubic(Resize.Resize):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = 'cubic'
        self.template_dict = {'1D':'./templates/layers/Resize/template_ResizeCubic1D.c.tpl',
                              '2D':'./templates/layers/Resize/template_ResizeCubic2D.c.tpl'}
    
    def forward_path_layer(self, input):
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        input= np.transpose(input,(1,2,0))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bicubic') #No numpy method for this layer
        output= np.transpose(output,(2,0,1))
        return self.activation_function.compute(output)
    
    #Compute the simple cubic convolution as describe in https://ieeexplore.ieee.org/document/1163711 (cf doc ONNX)
    def cubic_convolution_interpolation(self,f_1,f0,f1,f2,x,output):
        #give the values to the constants of the interpolation
        s = '                f_1 = ' + f_1 + ';\n'
        s+= '                f0 = ' + f0 + ';\n'
        s+= '                f1 = ' + f1 + ';\n'
        s+= '                f2 = ' + f2 + ';\n'
        s+= '                s = ' + x + '-floor('+x+');\n'
        s+= '                '+output+' = '
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
            mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[j + ' + str(self.output_width) + '*(i + ' + str(self.output_height) + '*f)]')

        mustach_hash['cubic_coeff_a'] = self.cubic_coeff_a
        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        
        if ((self.input_height == 1) and (self.input_width > 1)):
            mustach_hash['dimension'] = self.input_width
            mustach_hash['coordinate_transformation_mode'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x')
            dimension = '1D'
            #return self.write_cst_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            mustach_hash['dimension'] = self.input_height
            mustach_hash['coordinate_transformation_mode'] = self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x')
            dimension = '1D'
            #return self.write_cst_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
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

        return pystache.render(template, mustach_hash)