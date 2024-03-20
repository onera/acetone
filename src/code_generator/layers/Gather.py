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

#extract a list of subtensor from a given tensor
#attribut: axis alongside of which the submatrix will be extracted (if the desired submatrix must have the height, width or channels of the parent tensor)
#input: a tensor
#output: a list of tensor
class Gather(Layers.Layers):
    
    def __init__(self, idx, size, axis,  indices, input_shape, output_shape):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Gather'
        self.indices = indices
        self.axis = axis
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        
        
    def write_loops(self,source_file):
        flat_indices = np.ndarray.flatten(np.array(self.indices)) #may have indices like: [[0,1],[1,2]]
        source_file.write('    int indice['+str(len(flat_indices))+'] = {') #the list of the indices to use
        for i in flat_indices:
            source_file.write(str(i))
            if(i != flat_indices[-1]):
                source_file.write(", ")
            else:
                source_file.write("};\n")
        #the indice that will change depend on the axis. 
        if(self.axis == 1): #collecting channels
            source_file.write('    for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n    {\n')
            source_file.write('        int f = indice[k];\n')
            source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
            source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        if(self.axis == 2): #collecting heights
            source_file.write('    for (int f = 0; f < ' + str(self.input_channels) + '; f++)\n    {\n')
            source_file.write('        for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n        {\n')
            source_file.write('            int i = indice[k];\n')
            source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        if(self.axis == 3): #collecting widths
            source_file.write('    for (int f = 0; f < ' + str(self.input_channels) + '; f++)\n    {\n')
            source_file.write('        for (int i = 0; i < ' + str(self.input_height) + '; i++)\n        {\n')
            source_file.write('            for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n            {\n')
            source_file.write('                int j = indice[k];\n')
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    int position = 0;\n')#to know the position of the next element to add
        self.write_loops(source_file)
        source_file.write('                output_cur_'+str(self.road)+'[position]'+
                          ' = '+output_str+'[j + '+str(self.input_width)+' * (i + '+str(self.input_height)+' * f)];\n')
        source_file.write('                position++;\n')
        source_file.write('            }\n        }\n    }\n\n ')
        
    def feedforward(self,input):
        input = input.reshape(self.input_channels,self.input_height,self.input_width)
        return np.take(input, indices=self.indices, axis=self.axis-1)
