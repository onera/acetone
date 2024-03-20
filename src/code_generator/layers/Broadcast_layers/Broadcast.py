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
from abc import abstractmethod


#The class of the Layers which compute operation with broadcast numpy style
#attribut: none
#input: a list of tensor
#output: the resultant tensor
class Broadcast(Layers.Layers):
    def __init__(self, idx, size, input_shapes, output_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ''
        self.input_shapes = input_shapes
        
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]

        self.activation_function = activation_function
    
    @abstractmethod
    def specific_operator(self,source_file):
        pass

    def write_add_a_tensor(self,source_file): #a function to do the broadcast numpy style
        #Min, Max and Avg follow the rule of broadcasting but need a operator before the iteration of previous layer 
        if(self.name == 'Maximum'):
            source_file.write('max(')
        elif(self.name == 'Minimum'):
            source_file.write('min(')
        elif(self.name == 'Average'):
            source_file.write('(')
        #Going through all the ancestors 
        for k in range(len(self.previous_layer)):
            if(k!=0):
                self.specific_operator(source_file) #the operator to broadcats
            input_shape = self.input_shapes[k]
            output_str = self.previous_layer[k].output_str #if the value is in a road or saved eslewhere
            #The % operator allow to always saty in teh range of the tensor's indices 
            source_file.write(output_str+'[(j % '+ str(input_shape[3]) + ') + ' + str(input_shape[3]) + ' * ((i % ' + str(input_shape[2]) + ') + ' + str(input_shape[2]) + ' * (f % '+str(input_shape[1])+'))]' )
        
        #Closing the operator of Max,Min and Avg
        if(self.name == 'Maximum' or self.name=='Minimum'):
            source_file.write(')')
        elif(self.name=='Average'):
            source_file.write(')/('+ str(len(self.previous_layer)) +')\n')
        
        source_file.write(';\n')
    
    #Go through all the indices and do the operation
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n             {\n')
        source_file.write('                tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
        self.write_add_a_tensor(source_file)
        source_file.write('            }\n        }\n    }\n')
        a = self.activation_function.write_activation_str('tensor_temp[k]')
        source_file.write('    for (int k = 0; k < ' + str(self.size) + '; k++){\n        output_'+str(self.road)+'[k] = '+a+';\n    }\n\n')
    
    @abstractmethod
    def feedforward(self, inputs):
        pass