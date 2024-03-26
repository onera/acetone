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
import numpy as np

#Concatenate two tensor alongside a given axis
#attribut: axis alongside of which the concatenation will be done
#input: a list of tensor to concatenate
#output: the concatenated tensor 
class Concatenate(Layers.Layers):
    def __init__(self, idx, size, axis, input_shapes,output_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = 'Concatenate'
        self.axis = axis
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.activation_function = activation_function

    def write_concat(self, source_file):
        borne_sup = 0
        borne_inf = 0
        for k in range(len(self.previous_layer)):
            input_shape = self.input_shapes[k]
            output_str = self.previous_layer[k].output_str
            #We take the value of the matrix only if the indices are inside the adequat limits
            #the max indice is always at one length of matrix after the min indice
            
            if (self.axis == 1):
                #concat alongside the channels
                borne_sup += input_shape[1]
                source_file.write('                if((f < '+str(borne_sup) +') && (f >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[j  + ' + str(input_shape[3]) + ' * (i + ' + str(input_shape[2]) + ' * (f - ' + str(borne_inf) + ') )];\n                }\n')
                borne_inf += input_shape[1]
            if (self.axis == 2):
                #concat alongside the height
                borne_sup += input_shape[2]
                source_file.write('                if((i < '+str(borne_sup) +') && (i >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[j  + ' + str(input_shape[3]) + ' * ( (i - ' + str(borne_inf) + ') + ' + str(input_shape[2]) + ' * f )];\n                }\n')
                borne_inf += input_shape[2]
            if (self.axis == 3):
                #concat alongside the width
                borne_sup += input_shape[3]
                source_file.write('                if((j < '+str(borne_sup) +') && (j >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[(j - ' + str(borne_inf) + ') + ' + str(input_shape[3]) + ' * ( i + ' + str(input_shape[2]) + ' * f )];\n                }\n')
                borne_inf += input_shape[3]
            

            
    
    def write_to_function_source_file(self, source_file):
        #we go through all the indices of the tensor then write the opeartion depending on the axis
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.write_concat(source_file)
        source_file.write('            }\n        }\n    }\n\n')
        a = self.activation_function.write_activation_str('tensor_temp[k]')
        source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = '+a+';\n    }\n')

    def feedforward(self, inputs):
        output = inputs[0]
        output = output.reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        for i in range(1,len(inputs)):
            input = inputs[i]
            input = input.reshape(self.input_shapes[i][1],self.input_shapes[i][2],self.input_shapes[i][3])
            output = np.concatenate((output, input),axis=self.axis - 1) 
        return self.activation_function.compute(output)
