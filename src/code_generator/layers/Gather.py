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
import pystache

#extract a list of subtensor from a given tensor
#attribut: axis alongside of which the submatrix will be extracted (if the desired submatrix must have the height, width or channels of the parent tensor)
#input: a tensor
#output: a list of tensor
class Gather(Layers.Layers):
    
    def __init__(self, idx, size, axis,  indices, input_shape, output_shape,activation_function):
        
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
        self.activation_function = activation_function
        
        
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
    
    def write_to_function_source_file(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.road
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('tensor_temp[position]')

        mustach_hash['indices_len'] = len(self.indices.flatten())
        mustach_hash['input_width'] = self.input_width
        mustach_hash['input_height'] = self.input_height

        if(self.axis == 1):
            mustach_hash['channels'] = True
            mustach_hash['output_height'] = self.output_height
            mustach_hash['output_width'] = self.output_width
        elif(self.axis == 2):
            mustach_hash['heights'] = True
            mustach_hash['output_channels'] = self.input_channels
            mustach_hash['output_width'] = self.output_width
        elif(self.axis == 3):
            mustach_hash['widths'] = True
            mustach_hash['output_channels'] = self.input_channels
            mustach_hash['output_height'] = self.output_height

        if(self.activation_function.name == 'linear'):
            mustach_hash['linear'] = True

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('tensor_temp[position]',self.idx,'position')

        with open('src/templates/layers/template_Gather.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
        
    def feedforward(self,input):
        input = input.reshape(self.input_channels,self.input_height,self.input_width)
        return np.take(input, indices=self.indices, axis=self.axis-1)
