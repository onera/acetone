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

class InputLayer(Layers.Layers):

    def __init__(self, idx, size):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Input_layer'

    def write_to_function_source_file(self, source_file):
        
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        output_'+str(self.road)+'[i] = nn_input[i]; \n    } \n\n')

    def feedforward(self, input):
        
        return input 