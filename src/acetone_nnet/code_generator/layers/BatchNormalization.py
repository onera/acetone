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

from ..Layer import Layer
from ..activation_functions import ActivationFunctions
import numpy as np
import pystache

class BatchNormalization(Layer):

    def __init__(self, idx:int, size:int, input_shape:list, epsilon:float|int, scale:np.ndarray, biases:np.ndarray, mean:np.ndarray, var:np.ndarray, activation_function:ActivationFunctions):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'BatchNormalization'
        self.output_channels = input_shape[1]
        self.output_height = input_shape[2]
        self.output_width = input_shape[3]
        self.epsilon = epsilon
        self.scale = scale
        self.mean = mean
        self.var = var
        self.biases = biases
        self.nb_biases = self.count_elements_array(self.biases)

        self.activation_function = activation_function

        ####### Checking the instantiation#######

        ### Checking argument type ###
        assert type(self.idx) == int
        assert type(self.size) == int
        assert type(self.output_channels) == int
        assert type(self.output_height) == int
        assert type(self.output_width) == int
        assert type(self.epsilon) == float or type(self.epsilon) == int
        assert type(self.scale) == np.ndarray
        assert type(self.mean) == np.ndarray 
        assert type(self.var) == np.ndarray
        assert type(self.biases) == np.ndarray
        assert isinstance(self.activation_function, ActivationFunctions)

        ### Checking value consistency ###
        assert self.size == self.output_channels*self.output_height*self.output_width
        assert len(self.scale.shape) == 1 and self.scale.shape[0] == self.output_channels
        assert len(self.mean.shape) == 1 and self.mean.shape[0] == self.output_channels
        assert len(self.var.shape) == 1 and self.var.shape[0] == self.output_channels
        assert len(self.biases.shape) == 1 and self.biases.shape[0] == self.output_channels


    def generate_inference_code_layer(self):
        #Variable indicating under which name the input tensor is
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['path'] = self.path
        
        if(self.activation_function.name != 'linear'):
            mustach_hash['activation_function'] = self.activation_function.write_activation_str('output_'+str(self.path)+'[k + '+str(self.output_height*self.output_width)+'*f]')

        mustach_hash['input_channels'] = self.output_channels
        mustach_hash['channel_size'] = self.output_height*self.output_width
        mustach_hash['epsilon'] = self.epsilon

        with open(self.template_path+'layers/template_BatchNormalization.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def forward_path_layer(self, input:np.ndarray):
        input = np.reshape(input, (self.output_channels, self.output_height, self.output_width))
        output = []
        for i in range(self.output_channels):
            output.append((input[i] - self.mean[i])/np.sqrt(self.var[i] + self.epsilon)*self.scale[i] + self.biases[i])
        return np.array(output)