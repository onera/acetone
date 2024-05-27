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
import math
from abc import abstractmethod

class Conv2D(Layer):
    
    def __init__(self, idx:int, conv_algorithm:str, size:int, padding:str|np.ndarray, strides:int, kernel_h:int, kernel_w:int, dilation_rate:int, nb_filters:int, input_shape:list, output_shape:list, weights:np.ndarray, biases:np.ndarray, activation_function:ActivationFunctions):
        super().__init__()
        self.conv_algorithm = conv_algorithm
        self.idx = idx
        self.size = size
        self.name = 'Conv2D'
        self.padding = padding
        self.strides = strides
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
    
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]

        self.input_shape = [self.input_channels, self.input_height, self.input_width]
        self.output_channels = self.nb_filters

        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function
        self.local_var = 'sum'

        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)
        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.kernel_h, self.kernel_w, self.strides, self.dilation_rate)
    
        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Conv2D (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Conv2D (size must be int)")
        if type(conv_algorithm) != str:
            raise TypeError("Error: conv algorithm type in Conv2D (must be str)")
        if type(self.padding) != str and any(type(pad) != int for pad in self.padding):
            raise TypeError("Error: padding type in Conv2D (must be str or ints)")
        if  type(self.strides)!= int:
            raise TypeError("Error: strides type in Conv2D (must be int)")
        if  type(self.kernel_h)!= int:
            raise TypeError("Error: kernel_h type in Conv2D (must be int)")
        if  type(self.kernel_w)!= int:
            raise TypeError("Error: kernel_w type in Conv2D (must be int)")
        if  type(self.dilation_rate)!= int:
            raise TypeError("Error: dilation_rate type in Conv2D (must be int)")
        if  type(self.nb_filters)!= int:
            raise TypeError("Error: nb_filters type in Conv2D (must be int)")
        if type(self.input_channels) != int:
            raise TypeError("Error: input channels type in Conv2D (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in Conv2D (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in Conv2D (must be int)")
        if type(self.output_channels) != int:
            raise TypeError("Error: output channels type in Conv2D (must be int)")
        if type(self.output_height) != int:
            raise TypeError("Error: output height type in Conv2D (must be int)")
        if type(self.output_width) != int:
            raise TypeError("Error: output width type in Conv2D (must be int)")
        if type(self.weights) != np.ndarray:
            raise TypeError("Error: weights in Conv2D (weights must be an numpy array)")
        if type(self.biases) != np.ndarray:
            raise TypeError("Error: biases in Conv2D (biases must be an numpy array)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Conv2D (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in Conv2D ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        if self.weights.shape != (self.input_channels, self.kernel_h, self.kernel_w, self.nb_filters):
            raise ValueError("Error: non consistency between weight shape and operation parameters in Conv2D ("+str(self.weights.shape)+"!="+str((self.input_channels, self.kernel_h, self.kernel_w, self.nb_filters))+")")
        if len(self.biases.shape) != 1 or self.biases.shape[0] != self.nb_filters:
            raise ValueError("Error: non consistency between the var shape and the output shape in Conv2D ("+str(self.biases.shape)+"!="+str(self.nb_filters)+")")
        if self.output_height != math.floor((self.input_height + self.pad_bottom + self.pad_top - self.kernel_h - (self.kernel_h - 1)*(self.dilation_rate - 1))/self.strides) + 1:
            raise ValueError("Error: non consistency between the output height and the parameter of the operation in Conv2D ("+str(self.output_height)+"!="+str(math.floor((self.input_height + self.pad_bottom + self.pad_top - self.kernel_h - (self.kernel_h - 1)*(self.dilation_rate - 1))/self.strides) + 1)+")")
        if self.output_width != math.floor((self.input_width + self.pad_left + self.pad_right - self.kernel_w - (self.kernel_w - 1)*(self.dilation_rate - 1))/self.strides) + 1:
            raise ValueError("Error: non consistency between the output width and the parameter of the operation in Conv2D ("+str(self.output_width)+"!="+str(math.floor((self.input_width + self.pad_left + self.pad_right - self.kernel_w - (self.kernel_w - 1)*(self.dilation_rate - 1))/self.strides) + 1)+")")
        if self.conv_algorithm not in ['6loops',
                                       'indirect_gemm_nn','indirect_gemm_tn','indirect_gemm_nt','indirect_gemm_tt',
                                       'std_gemm_nn','std_gemm_tn','std_gemm_nt','std_gemm_']:
            raise ValueError("Error: conv algorithm value in Conv2D ("+self.conv_algorithm+")")

    @abstractmethod
    def generate_inference_code_layer(self):
        pass

    def forward_path_layer(self, input:np.ndarray):
        # Conv for chw
        input = input.reshape(self.input_channels, self.input_height, self.input_width)
        
        output = np.zeros((self.nb_filters, self.output_height, self.output_width))
        print(self.weights.shape)

        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input
            
        for f in range(self.nb_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                        output[f,i,j]=np.sum(input_padded[:, i*self.strides:i*self.strides+self.kernel_h, j*self.strides:j*self.strides+self.kernel_w] 
                                            * self.weights[:,:,:,f]) + self.biases[f]
        return self.activation_function.compute(output)