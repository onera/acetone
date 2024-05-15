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

#Return a tensor with where each position (f,i,j) contains the max of all the values at position (f,i,j) in each tensor
class Maximum(Broadcast):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = 'Maximum'
        self.specific_operator = ', '

    def forward_path_layer(self, inputs:np.ndarray):
        maxi = inputs[0]
        for input in inputs[1:]:
            maxi = np.maximum(maxi, input)
        return self.activation_function.compute(maxi)
