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

#The Edge mode of the Pad layers
#Pads with the edge values of array.
class Edge_pad(Pad.Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes, input_shape,activation_function)
        self.mode = 'edge'
    
    #Either go the the edge coordinates, or stay with the same coordinates
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write('                new_f =  f - '+self.pads[1]+';\n')#take the indice in the input tensor
        source_file.write('                new_i =  i - '+self.pads[2]+';\n')
        source_file.write('                new_j =  j - '+self.pads[3]+';\n')
        source_file.write(self.write_change_indice('f',self.pads[1],self.input_shape[1],'new_f'))#Going to an edge if necessary
        source_file.write(self.write_change_indice('i',self.pads[2],self.input_shape[2],'new_i'))
        source_file.write(self.write_change_indice('j',self.pads[3],self.input_shape[3],'new_j'))
        #Giving the value to the output tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
