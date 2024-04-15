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
import numpy as np
import pystache

class MatMul(Layer):

    def __init__(self, idx, size, input_shape, weights, side, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'MatMul'
        self.weights = np.asarray(weights)
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        self.side = side
        self.input_shape = input_shape
        self.nb_weights = self.count_elements_array(self.weights)

    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(self.local_var)

        mustach_hash['prev_size'] = self.previous_layer[0].size
        mustach_hash['side'] = self.side

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str(self.local_var,self.idx,'i')

            if(self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True
        
        with open(self.template_path+'layers/template_MatMul.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def forward_path_layer(self, input):
        
        input = input.reshape(self.previous_layer[0].size)
        if (self.side):
            weights = np.moveaxis(self.weights, 3,0)
            weights = np.reshape(weights, (weights.shape[1],weights.shape[2],weights.shape[3],weights.shape[0]))
            return self.activation_function.compute((np.matmul(weights,input)))
        else:
            return self.activation_function.compute((np.matmul(input,self.weights)))
