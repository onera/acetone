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
from..activation_functions import ActivationFunctions
import numpy as np
import pystache


#Do the dotproduct of two tensors
#atribut: the axis alongside of which the dot product will be done. if type(axis) == int, same axis for the two tensor. can be tuples of axis (element i of axis represent axis of tensor i)
#input: two tensor
#output: the resultant tensor 
class Dot(Layer):
    def __init__(self, idx:int, size:int, axis:int|list, input_shapes:list, output_shape:list, activation_function:ActivationFunctions):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = 'Dot'
        #we seek a tuple of axis
        if (type(axis) == int):
            self.axis = [axis,axis]
        else:
            self.axis = axis
        self.output_fourth_dim = output_shape[1]
        self.output_height = output_shape[3]
        self.output_width = output_shape[4]
        self.output_channels = output_shape[2]
        self.activation_function = activation_function            

    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['size'] = self.size
        mustach_hash['road'] = self.path

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')

        mustach_hash['output_fourth_dim'] = self.output_fourth_dim
        mustach_hash['output_channels'] = self.output_channels
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['axis_dim'] = self.input_shapes[0][self.axis[0]]
        mustach_hash['output_str_left'] = self.previous_layer[0].output_str
        mustach_hash['output_str_right'] = self.previous_layer[1].output_str

        if(self.axis[0] == 2):
            mustach_hash['indice_left'] = 'k + '+ str(self.input_shapes[0][self.axis[0]]) + ' * (f + ' + str(self.output_channels) + ' * p)'
        elif(self.axis[0] == 1):
            mustach_hash['indice_left'] = 'f + '+ str(self.output_channels) + ' * (k + ' + str(self.input_shapes[0][self.axis[0]]) + ' * p)'
        elif(self.axis[0] == 0):
            mustach_hash['indice_left'] = 'f + '+ str(self.output_channels) + ' * (p + ' + str(self.output_fourth_dim) + ' * k)'
        
        if(self.axis[1] == 2):
            mustach_hash['indice_right'] = 'k + '+ str(self.input_shapes[1][self.axis[1]]) + ' * (j + ' + str(self.output_width) + ' * i)'
        elif(self.axis[1] == 1):
            mustach_hash['indice_right'] = 'j + '+ str(self.output_width) + ' * (k + ' + str(self.input_shapes[1][self.axis[1]]) + ' * i)'
        elif(self.axis[1] == 0):
            mustach_hash['indice_right'] = 'j + '+ str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * k)'
        
        with open(self.template_path+'layers/template_Dot.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def forward_path_layer(self, inputs:np.ndarray):
        inputs[0] = inputs[0].reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        inputs[1] = inputs[1].reshape(self.input_shapes[1][1],self.input_shapes[1][2],self.input_shapes[1][3])
        output = np.tensordot(inputs[0],inputs[1],axes=[self.axis[0]-1,self.axis[1]-1])
        return self.activation_function.compute(output)
