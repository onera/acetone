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

import code_generator.layers.Layers as Layers
import numpy as np
from abc import abstractmethod

class Conv2D(Layers.Layers):
    
    def __init__(self, idx, conv_algorithm, data_format, size, padding, strides, kernel_h, kernel_w, dilation_rate, nb_filters, input_shape, output_shape, weights, biases, activation_function):
        super().__init__()
        self.conv_algorithm = conv_algorithm
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = 'Conv2D'
        self.padding = padding
        self.strides = strides
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        
        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]

        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'sum'

        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)
        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.kernel_h, self.kernel_w, self.strides, self.dilation_rate)
    
    @abstractmethod
    def write_to_function_source_file(self, source_file):
        pass

    def feedforward(self, input):
        # Conv for chw
        if(self.data_format == 'channels_last'):
            input = input.reshape(self.input_height, self.input_width, self.input_channels)
            input= np.transpose(input,(2,0,1))
            
        elif(self.data_format == 'channels_first'):
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
        if(self.data_format == 'channels_last'):
            output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)