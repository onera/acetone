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

import Pad

#The Reflect mode of the Pad layers
#Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
class Reflect_pad(Pad.Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes, input_shape,activation_function)
        self.mode = 'reflect'
    
    #A function to find the correponding coordinates in the input tensor
    def write_reflect(self,new_indice,indice_max,old_indice,pad):
        s = '                '+new_indice+' = ('+str(pad)+' - '+old_indice+') % (2 * '+str(indice_max)+');\n'
        s+= '                if('+new_indice+' > '+str(indice_max)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = 2 * '+str(indice_max)+' - '+new_indice+';\n'
        s+= '                }\n'
        return s
        
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write(self.write_reflect('new_f',self.input_shape[1]-1,'f',self.pads[1]))#Finding the correponding coordinates
        source_file.write(self.write_reflect('new_i',self.input_shape[2]-1,'i',self.pads[2]))
        source_file.write(self.write_reflect('new_j',self.input_shape[3]-1,'j',self.pads[3]))
        #Giving the value to the output tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
