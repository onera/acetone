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

import pystache
import numpy as np
from abc import ABC
from ... import templates

class Normalizer(ABC):

    def __init__(self, input_size:int, output_size:int, mins:list, maxes:list, means:list, ranges:list):
        self.input_size = input_size
        self.output_size = output_size
        self.mins = mins
        self.maxes = maxes
        self.means = means
        self.ranges = ranges
        self.template_path = templates.__file__[:-11]
        super().__init__()

    def array_to_str(self, array:list):
        s = "{"
        for element in array:
            s += str(element) + ", "
        s = s[:-2] + "}"
        return s

    def write_pre_processing(self):
        with open(self.template_path+'normalization/template_pre_processing.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template,{'input_size':self.input_size})


    def write_post_processing(self):
        with open(self.template_path+'normalization/template_post_processing.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template,{'output_size':self.output_size})

    def write_normalization_cst_in_header_file(self):
        mustach_hash= {}

        mustach_hash['input_size'] = self.input_size

        with open(self.template_path+'normalization/template_normalization_cst_in_header_file.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def write_normalization_cst_in_globalvars_file(self):
        mustach_hash= {}

        mustach_hash['input_size'] = self.input_size
        mustach_hash['input_min'] = self.array_to_str(self.mins)
        mustach_hash['input_max'] = self.array_to_str(self.maxes)
        mustach_hash['input_mean'] = self.array_to_str(self.means[:-1])
        mustach_hash['input_range'] = self.array_to_str(self.ranges[:-1])
        mustach_hash['output_mean'] = self.means[-1]
        mustach_hash['output_range'] = self.ranges[-1]

        with open(self.template_path+'normalization/template_normalization_cst_in_global_var_file.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return  pystache.render(template, mustach_hash)

    def pre_processing(self, nn_input:np.ndarray):
        inputs = nn_input.flatten()
        for i in range(self.input_size):
            inputs[i] = (max(self.mins[i],min(self.maxes[i],inputs[i])) - self.means[i]) / self.ranges[i]
        return np.reshape(inputs, nn_input.shape)

    def post_processing(self, nn_output:np.ndarray):
        for i in range(len(nn_output)):
            nn_output[i] = nn_output[i]*self.ranges[-1] + self.means[-1]
        return nn_output