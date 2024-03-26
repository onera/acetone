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

#Confine each element of the entry between two values
#attribut: a value min and a values max which define the intervalle of observation
#input: a tensor
#output: the resultant tensor
class Clip(Layers.Layers):
    def __init__(self, idx, size, max, min):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Clip'
        self.max = max
        self.min = min
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; i++)\n    {\n')#going through all the elements of the tensor
        source_file.write('        if('+output_str+'[i] < '+ str(self.min) +')\n        {\n')#if it's smaller than the min, it takes the min value
        source_file.write('            output_'+str(self.road)+'[i] = ' + str(self.min) + ';\n        }else{\n')
        source_file.write('            if('+output_str+'[i] > '+ str(self.max) +')\n            {\n')#if it's bigger than the max, it takes the max value
        source_file.write('                output_'+str(self.road)+'[i] = ' + str(self.max) + ';\n            }else{\n')
        source_file.write('                    output_'+str(self.road)+'[i] = '+output_str+'[i];\n            }\n')
        source_file.write('        }\n    }\n\n')

    def feedforward(self, input):
        return np.clip(input, self.min,self.max)