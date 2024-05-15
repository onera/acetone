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

#Subtraction of several tensors
class Subtract(Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Subtract'
        self.specific_operator = ' - '
    
    def forward_path_layer(self, inputs:np.ndarray):
        if(self.constant is None):
            constant = 0
        else: 
            constant = self.constant
        if(len(self.previous_layer) > 1):
            output = inputs[0]
            for input in inputs[1:]:
                output -= input
        else:
            output = inputs
        return self.activation_function.compute(output-constant)
