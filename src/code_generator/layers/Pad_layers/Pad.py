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

#The Pad Layers
#Pad alongside each dimmensions
#attribut: the mode of padding required
#input: a tensor to be padded, the desired pads, the value of teh constant if mode == constant
#output:the resized tensor
######################### cf https://onnx.ai/onnx/operators/onnx__Pad.html for the doc
class Pad(Layers.Layers):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.pads = pads
        self.constant_value = constant_value
        self.axes = axes
        self.name = 'Pad'
        self.input_shape = input_shape
        self.output_channels = input_shape[1] + pads[1] + pads[5]
        self.output_height = input_shape[2] + pads[2] + pads[6]
        self.output_width = input_shape[3] + pads[3] + pads[7]
        self.mode = ''
        self.activation_function = activation_function
    
    def feedforward(self, input):
        input = input.reshape(self.input_shape[1], self.input_shape[2], self.input_shape[3])
        nb_dim = len(self.pads)//2
        pad_width = [(self.pads[i],self.pads[i+nb_dim]) for i in range(1,nb_dim)] #Constructing the pads accordingly to the numpy nomenclature
        return self.activation_function.compute(np.pad(input,pad_width=pad_width,mode=self.mode,constant_values=self.constant_value,))
    
    @abstractmethod
    def write_padding(self,source_file):
        pass
    
    def write_to_function_source_file(self, source_file,):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.write_padding(source_file) #Depend on how the padding is done
        source_file.write('            }\n        }\n    }\n\n')
