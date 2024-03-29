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

import code_generator.layers.Broadcast_layers.Broadcast as Broadcast

#Division of several tensors
class Divide(Broadcast.Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Divide'
    
    def specific_operator(self,source_file):
        return source_file.write(' / ')
    
    def feedforward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output /= input
        return self.activation_function.compute(output)