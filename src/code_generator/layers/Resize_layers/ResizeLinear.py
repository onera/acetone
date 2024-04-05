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

#The mode Linear of the Resize layers
#The value in the output tensor are found thanks to a (bi)linear interpolation
class ResizeLinear(Resize.Resize):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = 'linear'
        self.template_dict = {'1D':'./src/templates/layers/Resize/template_ResizeLinear1D.c.tpl',
                              '2D':'./src/templates/layers/Resize/template_ResizeLinear2D.c.tpl'}
        
    def bilinear_interpolation(self):
        #the equation for the bilinear interpolation
        s = '(f11 * (x1 - x) * (y1 - y) +'
        s+= ' f21 * (x - x0) * (y1 - y) +'
        s+= ' f12 * (x1 - x) * (y - y0) +'
        s+= ' f22 * (x - x0) * (y - y0))' 
        s+= ' / ((x1 - x0) * (y1 - y0));\n'
        return s
    
    def write_cst_interpolation_2D(self):
        #the four points for a 2D interpolation and their values
        s = '    y1 = 0;\n'
        s+= '    y0 = '+str(self.input_height-1)+';\n'
        s+= '    x1 = '+str(self.input_width-1)+';\n'
        s+= '    x0 = 0;\n'
        return s
    
    #a function to write the point used in the interpolation
    def write_function_values_interpolation_2D(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[y0 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f12 = '+output_str+'[y0 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f22 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f21 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
        return s        
    
    #The function to do the interpolation but in 1D
    def write_cst_interpolation_1D_width(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the width of the tensor
        s = '    x1 = '+str(self.input_width-1)+';\n'
        s+= '    x0 = 0;\n'
        return s
    
    def write_function_values_interpolation_1D_width(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x0 + ' + str(self.input_width) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x1 + ' + str(self.input_width) + ' * f];\n'
        return s
    
    def write_cst_interpolation_1D_height(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the height of the tensor
        s = '    x1 = '+str(self.input_height-1)+';\n'
        s+= '    x0 = 0;\n'
        return s

    def write_function_values_interpolation_1D_height(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x0 + ' + str(self.input_height) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x1 + ' + str(self.input_height) + ' * f];\n'
        return s
    
    def linear_interpolation(self):
        #the equation for the interpolation
        s = '(f11 * (x1 - x) +'
        s+= ' f22 * (x - x0))' 
        s+= ' / (x1 - x0);\n'
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
    
    def write_to_function_source_file(self):

        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.road
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

        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write(self.write_cst_interpolation()) #The point used in the interpolation
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write(self.write_function_values_interpolation()) #f in the value of the element f_i_i
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
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        input= np.transpose(input,(1,2,0))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bilinear').numpy() #No numpy method for this layer
        output= np.transpose(output,(2,0,1))
        return self.activation_function.compute(output)
