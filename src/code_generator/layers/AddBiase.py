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
import pystache

class Add_Biass(Layers.Layers):

    def __init__(self, idx, size, biases,activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Add_Biass'
        self.biases = np.asarray(biases)
        self.nb_biases = self.count_elements_array(self.biases)
        self.activation_function = activation_function
    
    #Go through all the indices and do the operation
    def write_to_function_source_file(self):
        output_str = self.previous_layer[0].output_str#if the value is in a road or saved eslewhere

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['output_str'] = output_str
        mustach_hash['road'] = self.road
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output_'+str(self.road)+'[i]')

        if(self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        if(self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.road)+'[i]',self.idx,'i')

        with open('src/templates/layers/template_AddBiase.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def feedforward(self, input):
        input = input.reshape(self.previous_layer[0].size)

        return self.activation_function.compute(input + self.biases)