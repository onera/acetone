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

from ...Layer import Layer
from ...activation_functions import ActivationFunctions
import numpy as np
from abc import abstractmethod

#The Pad Layers
#Pad alongside each dimmensions
#attribut: the mode of padding required
#input: a tensor to be padded, the desired pads, the value of teh constant if mode == constant
#output:the resized tensor
######################### cf https://onnx.ai/onnx/operators/onnx__Pad.html for the doc
class Pad(Layer):
    
    def __init__(self, idx:int, size:int, pads:np.ndarray, constant_value:float, axes:np.ndarray|list, input_shape:list, activation_function:ActivationFunctions):
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

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Pad (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Pad (size must be int)")
        if any(type(pad) != int for pad in self.pads):
            raise TypeError("Error: pads type in Pad (must be int)")
        if type(self.constant_value) != float and type(self.constant_value) != int:
            raise TypeError("Error: constant value type in Pad (must be float or int)")
        if type(self.output_channels) != int:
            raise TypeError("Error: output channels type in Pad (must be int)")
        if type(self.output_height) != int:
            raise TypeError("Error: output height type in Pad (must be int)")
        if type(self.output_width) != int:
            raise TypeError("Error: output width type in Pad (must be int)")
        if any(type(shape) != int for shape in self.input_shape[1:]):
            raise TypeError("Error: input shape type in Pad (must be int)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Pad (activation function must be a sub-classe of acetone_nnet Activation Function)")


        ### Checking value consistency ###
        if self.size != self.output_channels*self.output_height*self.output_width:
            raise ValueError("Error: size value in Pad ("+str(self.size)+"!="+str(self.output_channels*self.output_height*self.output_width)+")")
        for axe in self.axes:
            if axe < 0 or axe >= 4:
                raise ValueError("Error: axe out of bound in Pad ("+str(axe)+"for tensor in 4 dimension with first dimension unused)")
    
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.input_shape[1], self.input_shape[2], self.input_shape[3])
        nb_dim = len(self.pads)//2
        pad_width = [(self.pads[i],self.pads[i+nb_dim]) for i in range(1,nb_dim)] #Constructing the pads accordingly to the numpy nomenclature
        return self.activation_function.compute(np.pad(input,pad_width=pad_width,mode=self.mode,constant_values=self.constant_value,))
    
    @abstractmethod
    def generate_inference_code_layer(self):
        pass