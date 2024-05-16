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

from ...Layer import Layer
from ...activation_functions import ActivationFunctions
import numpy as np
from abc import abstractmethod

#The resize Layers
#Only Height and Width are resized
#attribut: a lot of stuff
#input: a tensor to be resized, the desired size or a scale to multiply the size by, the region of interest
#output:the resized tensor
##############  https://onnx.ai/onnx/operators/onnx__Resize.html for more informations
#The strategie is always to go throught the elements, find the coordinate in the original tensor 
# and apply a transformation to the value in the original tensor to find the value to enter in the new tensor
class Resize(Layer):
    
    def __init__(self, idx:int, size:int, input_shape:list, activation_function:ActivationFunctions, axes:np.ndarray|list=[], coordinate_transformation_mode:str='half_pixel', exclude_outside:int=0,
                 keep_aspect_ratio_policy:str='stretch', boolean_resize:None|bool=None, target_size:np.ndarray|list=[], roi:np.ndarray|list=[], extrapolation_value:float|int=0., 
                 nearest_mode:str='round_prefer_floor', cubic_coeff_a:float=-0.75):
        super().__init__()
        self.idx = idx
        if(type(nearest_mode) == bytes):
            self.nearest_mode = str(nearest_mode)[2:-1]
        else: 
            self.nearest_mode = nearest_mode
        
        self.activation_function = activation_function
        self.cubic_coeff_a = cubic_coeff_a
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
        
        self.coordinate_transformation_mode_implem_mapping =  {"half_pixel":self.half_pixel_implem, 
                                                                "half_pixel_symmetric":self.half_pixel_symmetric_implem,
                                                                "pytorch_half_pixel":self.pytorch_half_pixel_implem,
                                                                "align_corners":self.align_corners_implem,
                                                                "asymmetric":self.asymmetric_implem,
                                                                "tf_crop_and_resize":self.tf_crop_and_resize_implem}
        
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.scale = [1.,0.,0.,0.]
        if (boolean_resize):
            self.scale = target_size
            self.output_channels = int(self.input_channels*target_size[1]) #should be equal to input_channels
            self.output_height = int(self.input_height*target_size[2])
            self.output_width = int(self.input_width*target_size[3])
        else:
            self.output_channels = target_size[1] #should be equal to input_channels
            self.output_height = target_size[2]
            self.output_width = target_size[3]
            self.scale[1] = self.output_channels / self.input_channels #should be 1
            self.scale[2] = self.output_height / self.input_height
            self.scale[3] = self.output_width / self.input_width

        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.idx) == int
        assert type(self.size) == int
        assert type(self.output_channels) == int
        assert type(self.output_height) == int
        assert type(self.output_width) == int
        assert type(self.input_channels) == int
        assert type(self.input_height) == int
        assert type(self.input_width) == int
        assert type(self.coordinate_transformation_mode) == str
        assert type(axes) == np.ndarray or type(axes) == list
        assert type(self.exclude_outside) == int
        assert type(self.keep_aspect_ratio_policy) == str
        assert type(self.roi) == np.ndarray or type(self.roi) == list
        assert type(self.extrapolation_value) == float or type(self.extrapolation_value) == int

        ### Checking value consistency ###
        assert self.size == self.output_channels*self.output_height*self.output_width
        assert self.coordinate_transformation_mode in ['half_pixel','half_pixel_symmetric','pytorch_half_pixel','align_corners','asymmetric','tf_crop_and_resize']
        assert self.keep_aspect_ratio_policy in ['stretch','not_larger','not_smaller']
        assert self.exclude_outside in [0,1]
        assert all(axe >=0 and axe < 4 for axe in self.axes)
        if self.roi:
            assert len(self.roi) == 2*len(self.axes) if self.axes else len(self.roi) == 8
            assert all(0 <= indice and indice <= 1 for indice in self.roi)
        assert all(0 < coeff for coeff in self.scale)
        

    @abstractmethod
    def forward_path_layer(self, input:np.ndarray):
        pass
    
    @abstractmethod
    def generate_inference_code_layer(self):
        pass

    #Defining the several coordinate transformations. cf documentation 
    def half_pixel(self, coord_resized:str, coord_dim,coord_original:str):
        s = coord_original + ' = ('+ coord_resized+' + 0.5)/'+ str(self.scale[coord_dim])+' - 0.5;'
        return s

    def half_pixel_implem(self, coordinate:int, coordinate_dimension:int):
        return (coordinate + 0.5)/self.scale[coordinate_dimension] - 0.5 
    
    def half_pixel_symmetric(self, coord_resized:str, coord_dim,coord_original:str):
        if(coord_dim==2):
            target_length = self.output_height*self.scale[2]
            input_length = self.input_height
        else: 
            target_length = self.output_width*self.scale[3]
            input_length = self.input_width

        s = 'float adjustment = ' + str(int(target_length)) + '/' + str(target_length)  +';\n'
        s += '                float center = ' + str(input_length) + '/2;\n'
        s += '                float offset = center*(1 - adjustment);\n'
        s +='                '+ coord_original + ' = offset + ('+ coord_resized+' + 0.5)/'+ str(self.scale[coord_dim])+' - 0.5;'
        return s
    
    def half_pixel_symmetric_implem(self, coordinate:int, coordinate_dimension:int):
        if(coordinate_dimension==2):
            target_length = self.output_height*self.scale[2]
            input_length = self.input_height
        else: 
            target_length = self.output_width*self.scale[3]
            input_length = self.input_width

        adjustment = int(target_length)/target_length
        center = input_length/2
        offset = center*(1 - adjustment)
        return offset + (coordinate + 0.5)/self.scale[coordinate_dimension] - 0.5 

    def pytorch_half_pixel(self, coord_resized:str, coord_dim,coord_original:str):
        if(coord_dim==2):
            target_length = self.output_height
        else: 
            target_length = self.output_width
        
        s = coord_original + ' = '
        if (target_length > 1):
            s += '('+ coord_resized+' + 0.5)/'+ str(self.scale[coord_dim])+' - 0.5;'
        else:
            s += '0;' 
        return s
    
    def pytorch_half_pixel_implem(self, coordinate:int, coordinate_dimension:int):
        if(coordinate_dimension==2):
            target_length = self.output_height
        else: 
            target_length = self.output_width
        
        if(target_length > 1):
            return (coordinate + 0.5)/self.scale[coordinate_dimension] - 0.5 
        else:
            return 0

    def align_corners(self, coord_resized:str, coord_dim,coord_original:str):
        if(coord_dim==2):
            length_original = self.input_height
            length_resized = self.output_height
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            
        s = coord_original + ' = ' +coord_resized+'*(' + str(length_original)+' - 1)/(' + str(length_resized)+' - 1);'
        return s
    
    def align_corners_implem(self, coordinate:int, coordinate_dimension:int):
        if(coordinate_dimension==2):
            length_original = self.input_height
            length_resized = self.output_height
        else: 
            length_original = self.input_width
            length_resized = self.output_width
        
        return coordinate * (length_original - 1)/(length_resized - 1)
    
    def asymmetric(self, coord_resized:str, coord_dim,coord_original:str):
        s = coord_original + ' = '+ coord_resized+'/'+ str(self.scale[coord_dim]) +';'
        return s
    
    def asymmetric_implem(self, coordinate:int, coordinate_dimension:int):
        return coordinate/self.scale[coordinate_dimension]
    
    def tf_crop_and_resize(self, coord_resized:str, coord_dim,coord_original:str):
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
        
        s = coord_original + ' = ' 
        if(length_resized > 1):
            s+= str(start) + '*('+ str(length_original)+' - 1) + '+ coord_resized+'*(' +str(end)+' - '+str(start)+')*('+str(length_original)+' - 1)/(' + str(length_resized)+' - 1);'
        else:
            s+= '0.5*(' +str(end)+' - '+str(start)+')*('+str(length_original)+' - 1);\n'
        s+= '                if(('+coord_original+' < 0) || ('+coord_original+' > '+str(length_original)+'){'+coord_original+' = '+ str(self.extrapolation_value)+'}'
        return s

    def tf_crop_and_resize_implem(self, coordinate:int, coordinate_dimension:int):
        if(coordinate_dimension==2):
            length_original = self.input_height
            length_resized = self.output_height
            start = self.roi[2]
            end = self.roi[6]
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            start = self.roi[3]
            end = self.roi[7]

        if(length_resized > 1):
            coordinate_original =  start*(length_original - 1) + coordinate*(end - start)*(length_original -1)/(length_resized - 1)
        else:
            coordinate_original = 0.5*(start + end)*(length_original - 1)

        if(coordinate_original >= length_original or coordinate_original < 0):
            return self.extrapolation_value
        else:
            return coordinate_original
