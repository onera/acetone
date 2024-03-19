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

import Layers
import numpy as np

class Softmax(Layers):

    def __init__(self, idx, size):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'

    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    sum = 0;\n\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; ++i)\n')
        source_file.write('        sum += exp('+output_str+'[i]);\n\n')
        source_file.write('    for (int j = 0; j < ' + str(self.size) + '; ++j)\n    {\n')
        source_file.write('        output_'+str(self.road)+'[j] = exp('+output_str+'[j])/sum;\n\n')
        if (self.fused_layer):
            b = self.fused_layer.write_activation_str('output_'+str(self.road)+'[j]', self.idx, 'j')
            source_file.write('        output_'+str(self.road)+'[j] = '+ b +';\n')
        source_file.write('    }\n\n')

    def feedforward(self, input):
        
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp)

        return output