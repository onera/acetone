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
from abc import abstractmethod

class Pooling2D(Layers.Layers):
    def __init__(self, idx, data_format, size, padding, strides, pool_size, input_shape, output_shape,**kwargs):
        
        super().__init__()
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = ''
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

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

        self.pooling_funtion = ''
        self.local_var = ''
        self.local_var_2 = ''
        self.output_var = ''

        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.pool_size,self.pool_size, self.strides)

    @abstractmethod    
    def specific_function(self, index, input_of_layer):
        pass

    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        if(self.data_format == 'channels_first'):
            indice = 'jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)'
        elif(self.data_format == 'channels_last'):
            indice = 'c + '+str(self.input_channels)+'*(jj + '+str(self.input_width)+'*ii)'
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')

        source_file.write('            ' + self.update_local_vars())

        source_file.write('                for (int m = 0; m < '+str(self.pool_size)+'; ++m)\n                {\n')
        source_file.write('                    for (int n = 0; n < '+str(self.pool_size)+'; ++n)\n                    {\n')
        source_file.write('                        int ii = i*'+str(self.strides)+' + m - '+str(self.pad_left)+';\n')
        source_file.write('                        int jj = j*'+str(self.strides)+' + n - '+str(self.pad_top)+';\n\n')
        source_file.write('                        if (ii >= 0 && ii < '+str( self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                        {\n')
        source_file.write(self.specific_function(indice, output_str))
        source_file.write('                        }\n                    }\n                }\n')
        
        if (self.fused_layer):
            b = self.fused_layer.write_activation_str(self.output_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)')
        else: 
            b = self.output_var

        source_file.write('            output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)]'+' = '+ b +';\n')
        
        source_file.write('            }\n        }\n    }\n\n')

    def feedforward(self, input):
        if(self.data_format == 'channels_last'):
            input = input.reshape(self.input_height, self.input_width, self.input_channels)
            input= np.transpose(input,(2,0,1))
            
        elif(self.data_format == 'channels_first'):
            input = input.reshape(self.input_channels, self.input_height, self.input_width)
            
        
        output = np.zeros((self.input_channels, self.output_height, self.output_width))
                
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input

        for c in range(self.input_channels):
            for i in range(self.output_height):
                for j in range(self.output_width): 
                    output[c,i,j]= self.pooling_function((input_padded[c, i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]))
        
        if(self.data_format == 'channels_last'):
            output= np.transpose(output,(1,2,0))
        return output
