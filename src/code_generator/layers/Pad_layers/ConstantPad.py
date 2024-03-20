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

import code_generator.layers.Pad_layers.Pad as Pad

#The Constant mode of the Pad layers
#Use a constant to fill paddings
class Constant_Pad(Pad.Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes,input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes,input_shape,activation_function)
        self.mode = 'constant'
    
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[(j -'+str(self.pads[3])+') + ' + str(self.input_shape[3]) + ' * ((i -'+str(self.pads[2])+') + ' + str(self.input_shape[2]) + ' * f)]')
        b = self.activation_function.write_activation_str(str(self.constant_value))
        #if the new indice is in the the old tensor, we take the value from the tensor

        source_file.write('                if((f >= '+str(self.pads[1])+') && (f <'+str(self.output_channels - self.pads[5])+') && (i >= '+str(self.pads[2])+') && (i <'+str(self.output_height - self.pads[6])+') && (j >= '+str(self.pads[3])+') && (j <'+str(self.output_width - self.pads[7])+'))\n')
        source_file.write('                {\n')
        source_file.write('                    output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        source_file.write('                }\n                else\n')#else, we take the constant value
        source_file.write('                {\n')
        source_file.write('                    output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+b+';\n')
        source_file.write('                }\n')
