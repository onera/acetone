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

from .Conv2D import Conv2D
import numpy as np
import pystache

class Conv2D_gemm(Conv2D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patches_height = self.input_channels * self.kernel_h * self.kernel_w
        self.patches_width = self.output_height * self.output_width
        self.patches_size = self.patches_height * self.patches_width

        self.conv_algorithm = self.conv_algorithm[-7:]
        self.algo_gemm_mapping = { 'gemm_nn' : self.write_gemm_nn,
                                   'gemm_nt' : self.write_gemm_nt,
                                   'gemm_tn' : self.write_gemm_tn,
                                   'gemm_tt' : self.write_gemm_tt}

    def write_gemm_nn(self, m:int, n:int, k:int, A:str, B:str, C:str, direct:bool):

        mustach_hash = {}

        mustach_hash['direct'] = direct
        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['ldA'] = k
        mustach_hash['B'] = B
        mustach_hash['ldB'] = n
        mustach_hash['C'] = C
        mustach_hash['ldC'] = n
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Conv/template_Conv_gemm_nn.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def write_gemm_nt(self, m:int, n:int, k:int, A:str, B:str, C:str, direct:bool):

        mustach_hash = {}

        mustach_hash['direct'] = direct
        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['ldA'] = k
        mustach_hash['B'] = B
        mustach_hash['ldB'] = n
        mustach_hash['C'] = C
        mustach_hash['ldC'] = n
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Conv/template_Conv_gemm_nt.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tn(self, m:int, n:int, k:int, A:str, B:str, C:str, direct:bool):

        mustach_hash = {}

        mustach_hash['direct'] = direct
        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['ldA'] = k
        mustach_hash['B'] = B
        mustach_hash['ldB'] = n
        mustach_hash['C'] = C
        mustach_hash['ldC'] = n
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Conv/template_Conv_gemm_tn.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tt(self, m:int, n:int, k:int, A:str, B:str, C:str, direct:bool):

        mustach_hash = {}

        mustach_hash['direct'] = direct
        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['ldA'] = k
        mustach_hash['B'] = B
        mustach_hash['ldB'] = n
        mustach_hash['C'] = C
        mustach_hash['ldC'] = n
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('sum')
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.path),self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Conv/template_Conv_gemm_tt.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)