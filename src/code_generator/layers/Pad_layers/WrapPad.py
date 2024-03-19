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

#The Wrap mode of the Pad layers
#Pads with the wrap of the vector along the axis. 
#The first values are used to pad the end and the end values are used to pad the beginning.
class Wrap_pad(Pad.Pad):
    
    def __init__(self, idx, size, mode, pads, constant_value, axes, input_shape):
        super().__init__(idx, size, mode, pads, constant_value, axes, input_shape)
        self.mode = 'wrap'
    
    #Translate to the value used to wrap the input tensor
    def write_change_indice(self,old_indice,pad,dim,new_indice):
        s = '                if(('+old_indice+' < '+str(pad)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = ('+new_indice+') % ('+str(dim)+');\n'
        s+= '                }\n'
        s+= '                if(('+old_indice+' >= '+str(pad + dim)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = ('+new_indice+') % ('+str(dim)+');\n'
        s+= '                }\n'
        return s
        
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write('                new_f =  f - '+self.pads[1]+';\n')#take the indice in the input tensor
        source_file.write('                new_i =  i - '+self.pads[2]+';\n')
        source_file.write('                new_j =  f - '+self.pads[3]+';\n')
        source_file.write(self.write_change_indice('f',self.pads[1],self.input_shape[1],'new_f'))#If necessary, taking the wrapping value
        source_file.write(self.write_change_indice('i',self.pads[2],self.input_shape[2],'new_i'))
        source_file.write(self.write_change_indice('j',self.pads[3],self.input_shape[3],'new_j'))
        #Giving the value to the output tensor
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
