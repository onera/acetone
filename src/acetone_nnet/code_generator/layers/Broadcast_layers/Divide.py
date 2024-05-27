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

from .Broadcast import Broadcast
import numpy as np

#Division of several tensors
class Divide(Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Divide'
        self.specific_operator = '/'
    
    def forward_path_layer(self, inputs:np.ndarray):
        if(self.constant is None):
            constant = np.ones(1)
        else: 
            constant = np.reshape(self.constant,self.input_shapes[-1][1:])
            self.input_shapes = np.delete(self.input_shapes,-1,axis=0)
        if(len(self.previous_layer) > 1):
            output = np.copy(inputs[0]).reshape(self.input_shapes[0][1:])
            for i in range(1,len(inputs)):
                output /= np.reshape(inputs[i],self.input_shapes[i][1:])
        else:
            output = np.reshape(inputs,self.input_shapes[0][1:])
        return self.activation_function.compute(output/constant)