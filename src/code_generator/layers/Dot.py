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


#Do the dotproduct of two tensors
#atribut: the axis alongside of which the dot product will be done. if type(axis) == int, same axis for the two tensor. can be tuples of axis (element i of axis represent axis of tensor i)
#input: two tensor
#output: the resultant tensor 
class Dot(Layers.Layers):
    def __init__(self, idx, size, axis, input_shapes,output_shape,activation_function):
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
    
    def write_dot(self,source_file, i):
        #depend if it's the first or the second tensor on the dot 
        if (i == 0):
            var = ['f','g']
            size = [self.output_channels,self.output_fourth_dim]
        else:
            var = ['j','i']
            size = [self.output_width,self.output_height]
        
        output_str = self.previous_layer[i].output_str
        source_file.write(output_str + '[')
        #the k indice correspond to the interation in the dot product, and it's position depend of the axis of the product.
        if (self.axis[i] == 2):
            source_file.write('k + '+ str(self.input_shapes[i][self.axis[i]]) + ' * ('+ var[0] +' + ' + str(size[0]) + ' * ' + var[1] + ')]')
        elif(self.axis[i] == 1):
            source_file.write(var[0] + ' + '+ str(size[0]) + ' * (k + ' + str(self.input_shapes[i][self.axis[i]]) + ' * ' + var[1] + ')]')
        elif(self.axis[i] == 0):
            source_file.write(var[0] + ' + '+ str(size[0]) + ' * ('+ var[1] +' + ' + str(size[1]) + ' * k)]')

            

    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for(int g = 0; g < ' + str(self.output_fourth_dim) + '; g++)\n    {\n')
        source_file.write('        for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n        {\n')#the two non variant dim of the first tensor
        source_file.write('            for (int i = 0; i < ' + str(self.output_height) + '; i++)\n            {\n')
        source_file.write('                for (int j = 0; j < ' + str(self.output_width) + '; j++)\n                {\n')#the two non variant dim of the second tensor
        source_file.write('                    register float output = 0;\n')
        source_file.write('                    for (int k = 0; k < ' + str(self.input_shapes[0][self.axis[0]]) + '; k++)\n                    {\n')
        source_file.write('                        output +=')
        self.write_dot(source_file,0)
        source_file.write(' * ')
        self.write_dot(source_file,1)
        #Apply the activation function and/or the fused function to the output of the layer
        a = self.activation_function.write_activation_str('output')
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * (f + ' + str(self.output_channels) +' * g))] = '+ a +';\n')
        source_file.write(';\n')
        source_file.write('                    }\n                }\n            }\n        }\n    }\n\n')
    
    def feedforward(self, inputs):
        inputs[0] = inputs[0].reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        inputs[1] = inputs[1].reshape(self.input_shapes[1][1],self.input_shapes[1][2],self.input_shapes[1][3])
        return self.activation_function.compute(np.tensordot(inputs[0],inputs[1],axes=[self.axis[0]-1,self.axis[1]-1]))
