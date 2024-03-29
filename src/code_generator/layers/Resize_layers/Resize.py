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
from abc import abstractmethod

#The resize Layers
#Only Height and Width are resized
#attribut: a lot of stuff
#input: a tensor to be resized, the desired size or a scale to multiply the size by, the region of interest
#output:the resized tensor
##############  https://onnx.ai/onnx/operators/onnx__Resize.html for more informations
#The strategie is always to go throught the elements, find the coordinate in the original tensor 
# and apply a transformation to the value in the original tensor to find the value to enter in the new tensor
class Resize(Layers.Layers):
    
    def __init__(self,idx,size,input_shape,axes,coordinate_transformation_mode,exclude_outside,
                 keep_aspect_ratio_policy,boolean_resize,target_size,roi,extrapolation_value, 
                 nearest_mode,activation_function):
        super().__init__()
        self.idx = idx
        if(type(nearest_mode) == bytes):
            self.nearest_mode = str(nearest_mode)[2:-1]
        else: 
            self.nearest_mode = nearest_mode
        
        self.activation_function = activation_function
        self.size = size
        self.axes = axes
        self.exclude_outside = exclude_outside
        self.keep_aspect_ratio_policy = keep_aspect_ratio_policy
        self.roi = roi
        self.name = 'Resize'
        self.extrapolation_value = extrapolation_value
        if(type(coordinate_transformation_mode) == bytes):
            self.coordinate_transformation_mode = str(coordinate_transformation_mode)[2:-1]
        else: 
            self.coordinate_transformation_mode = coordinate_transformation_mode
        self.coordinate_transformation_mode_mapping = {"half_pixel":self.half_pixel, 
                                                       "half_pixel_symmetric":self.half_pixel_symmetric,
                                                       "pytorch_half_pixel":self.pytorch_half_pixel,
                                                       "align_corners":self.align_corners,
                                                       "asymmetric":self.asymmetric,
                                                       "tf_crop_and_resize":self.tf_crop_and_resize}
        #if channel first
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        if (boolean_resize):
            self.scale = target_size
            self.output_channels = int(self.input_channels*target_size[1])
            self.output_height = int(self.input_height*target_size[2])
            self.output_width = int(self.input_width*target_size[3])
        else:
            self.output_channels = target_size[1]
            self.output_height = target_size[2]
            self.output_width = target_size[3]
            self.scale[1] = self.output_channels / self.input_channels
            self.scale[2] = self.output_height / self.input_height
            self.scale[3] = self.output_width / self.input_width
            
    @abstractmethod
    def feedforward(self,input):
        pass
    
    @abstractmethod
    def write_to_function_source_file(self, source_file):
        pass

    #Defining the several coordinate transformations. cf documentation 
    def half_pixel(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += coord_original + ' = ('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        return s
    
    def half_pixel_symmetric(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += 'float adjustment = ' + str(int(self.output_width)) + ' / ' + str(self.output_width)  +';\n'
        s += '                float center = ' + str(self.input_width) + ' / 2;\n'
        s += '                float offset = center * (1 - adjustment);\n'
        s +='                '+ coord_original + ' = offset + ('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        return s
    
    def pytorch_half_pixel(self,coord_resized,coord_dim,coord_original):
        s = '                '
        if(coord_dim==2):
            length = self.output_height
        else: length = self.output_width
        s += coord_original + ' = '
        if (length > 1):
            s += '('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        else:
            s += '0;\n' 
        return s
    
    def align_corners(self,coord_resized,coord_dim,coord_original):
        s = '                '
        if(coord_dim==2):
            length_original = self.input_height
            length_resized = self.output_height
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            
        s += coord_original + ' = ' +coord_resized+' * (' + str(length_original)+' - 1) / (' + str(length_resized)+' - 1);\n'
        return s
    
    def asymmetric(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += coord_original + ' = '+ coord_resized+' / '+ str(self.scale[coord_dim]) +';\n'
        return s
    
    def tf_crop_and_resize(self,coord_resized,coord_dim,coord_original):
        if(coord_dim==2):
            length_original = self.input_height
            length_resized = self.output_height
            start = self.roi[2]
            end = self.roi[6]
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            start = self.roi[3]
            end = self.roi[7]
        
        s = '                '
        s += coord_original + ' = ' 
        if(length_resized > 1):
            s+= str(start) + ' * ('+ str(length_original)+' - 1) + '+ coord_resized+' * (' +str(end)+' - '+str(start)+') * ('+str(length_original)+' - 1) / (' + str(length_resized)+' - 1);\n'
        else:
            s+= '0.5 * (' +str(end)+' - '+str(start)+') * ('+str(length_original)+' - 1);\n'
        s+= '                if(('+coord_original+' < 0) || ('+coord_original+' > '+str(length_original)+'){'+coord_original+' = '+ str(self.extrapolation_value)+'}\n'
        return s
