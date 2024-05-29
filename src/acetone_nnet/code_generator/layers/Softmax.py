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

from ..Layer import Layer
import numpy as np
import pystache

class Softmax(Layer):

    def __init__(self, idx:int, size:int, axis:int|None):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'
        self.axis = axis

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Softmax (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Softmax (size must be int)")
        if  type(self.axis)!= int and self.axis!=None:
            raise TypeError("Error: axis type in Softmax (axis must be int or None)")

    def generate_inference_code_layer(self):
        output_str = self.previous_layer[0].output_str

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['size'] = self.size
        mustach_hash['road'] = self.path
        mustach_hash['output_str'] = output_str

        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.path)+'[j]', self.idx, 'j')

        with open(self.template_path+'layers/template_Softmax.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def forward_path_layer(self, input:np.ndarray):
        if len(input.shape) == 3:
            input = np.expand_dims(input, 0)
            
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp, keepdims=1, axis=self.axis)

        return output