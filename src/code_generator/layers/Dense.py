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

class Dense(Layers.Layers):

    def __init__(self, idx, size, weights, biases, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Dense'
        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        
    def write_to_function_source_file(self, source_file):
        #Variable indicating under which name the input tensor is
        output_str = self.previous_layer[0].output_str
        
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        dotproduct = 0;\n')
        source_file.write( '        for (int j = 0; j < ' + str(self.previous_layer[0].size) + '; ++j)\n        {\n')
        source_file.write( '            dotproduct += '+output_str+'[j] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[(j + ' + str(self.previous_layer[0].size) + '*i)];\n        }\n')
        source_file.write( '        dotproduct += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[i];\n')
        
        self.write_activation_and_fusion(source_file, 'tensor_temp[i]', 'i', self.local_var,'        ')
        source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = tensor_temp[k];\n    }\n\n')

    def feedforward(self, input):
        input = input.reshape(self.previous_layer[0].size)
        return self.activation_function.compute(np.dot(input, self.weights) + self.biases)
