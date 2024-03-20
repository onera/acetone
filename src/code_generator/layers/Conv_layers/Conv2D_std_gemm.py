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

import code_generator.layers.Conv_layers.Conv2D as Conv2D
import numpy as np

class Conv2D_std_gemm(Conv2D.Conv2D):
    
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
        
        self.algo_patch_building_mapping = { 'gemm_nn' : self.write_im2col,
                                             'gemm_nt' : self.write_im2row,
                                             'gemm_tn' : self.write_im2col,
                                             'gemm_tt' : self.write_im2row}

    def write_im2col(self):
        output_str = self.previous_layer[0].output_str
        if('output' in output_str):
            output_str = 'tensor_temp'
        
        if(self.data_format == 'channels_first'):
            indice = '(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj'
        elif(self.data_format == 'channels_last'):
            indice = '(ii*'+str(self.input_width)+' + jj)*'+str(self.input_channels)+' + c_offset'

        s = '    // im2col\n'
        s+= '    for (int i = 0; i < '+str(self.patches_height)+'; ++i) {\n\n'
        s+= '        int i_offset = (i / '+str(self.kernel_w)+') % '+str(self.kernel_h)+';\n'
        s+= '        int j_offset = i % '+str(self.kernel_w)+';\n'
        s+= '        int c_offset = i / '+str(self.kernel_h)+' / '+str(self.kernel_w)+';\n\n'
        s+= '        for (int h = 0; h < '+str(self.output_height)+'; ++h) {\n'
        s+= '            for (int w = 0; w < '+str(self.output_width)+'; ++w) {\n\n'
        s+= '                int ii = h * '+str(self.strides)+' - '+str(self.pad_top)+' + i_offset; \n'
        s+= '                int jj = w * '+str(self.strides)+' - '+str(self.pad_left)+' + j_offset;\n\n'
        s+= '                int j = h*'+str(self.output_width)+' + w;\n'
        s+= '                if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n'
        s+= '                    output_'+str(self.road)+'[i*'+str(self.patches_width)+' + j] = '+output_str+'['+indice+'];\n'
        s+= '                else\n'
        s+= '                    output_'+str(self.road)+'[i*'+str(self.patches_width)+' + j] = 0;\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '    }\n'
        s+= '    \n\n'
        
        return s
    
    def write_im2row(self):
        output_str = self.previous_layer[0].output_str
        if('output' in output_str):
            output_str = 'tensor_temp'
        
        if(self.data_format == 'channels_first'):
            indice = '(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj'
        elif(self.data_format == 'channels_last'):
            indice = '(ii*'+str(self.input_width)+' + jj)*'+str(self.input_channels)+' + c_offset'
            
        s = '    // im2row\n'
        s+= '    for (int i = 0; i < '+str(self.patches_height)+'; ++i) {\n\n'
        s+= '        int i_offset = (i / '+str(self.kernel_w)+') % '+str(self.kernel_h)+';\n'
        s+= '        int j_offset = i % '+str(self.kernel_w)+';\n'
        s+= '        int c_offset = i / '+str(self.kernel_h)+' / '+str(self.kernel_w)+';\n\n'
        s+= '        for (int h = 0; h < '+str(self.output_height)+'; ++h) {\n'
        s+= '            for (int w = 0; w < '+str(self.output_width)+'; ++w) {\n\n'
        s+= '                int ii = h * '+str(self.strides)+' - '+str(self.pad_top)+' + i_offset; \n'
        s+= '                int jj = w * '+str(self.strides)+' - '+str(self.pad_left)+' + j_offset;\n\n'
        s+= '                int j = w*'+str(self.output_height)+' + h;\n'
        s+= '                if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n'
        s+= '                    output_'+str(self.road)+'[j*'+str(self.patches_height)+' + i] = '+output_str+'['+indice+'];\n'
        s+= '                else\n'
        s+= '                    output_'+str(self.road)+'[j*'+str(self.patches_height)+' + i] = 0;\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '    }\n'
        s+= '    \n\n'
        
        return s

    def write_gemm_nn(self, m, n, k, A, B):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = n
        self.ldC = n
        a = self.activation_function.write_activation_str('output')
        s = '    // gemm_nn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           register float weight = '+str(self.A)+'[i*'+str(self.ldA)+'+p];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               tensor_temp[i*'+str(self.ldC)+' + j] += weight * '+str(self.B)+'[p*'+str(self.ldB)+' + j];\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '        for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '            register float output = tensor_temp[i*'+str(self.ldC)+' + j];\n'
        s+= '            output += biases_' + self.name + '_' + str('{:02d}'.format(self.idx))+'[i];\n'
        if(self.fused_layer):
            if(self.activation_function.name != 'linear'):
                s+= '            output = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+a+';\n'
        s+= '        }\n'
        s+= '    }\n\n'        
        return s
    
    def write_gemm_nt(self, m, n, k, A, B):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = k
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register output = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               output += '+str(self.A)+'[i*'+str(self.ldA)+'+p] * '+str(self.B)+'[j*'+str(self.ldB)+' + p];\n'
        s+= '           }\n'
        s+= '           output += biases_'+ self.name + '_' + str("{:02d}".format(self.idx))+'[i];\n'
        if(self.fused_layer):
            if(self.activation_function.name != 'linear'):
                s+= '            output = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+a+';\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tn(self, m, n, k, A, B):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = n
        self.ldC = n
        a = self.activation_function.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]')

        s = '    // gemm_tn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           float register weight = '+str(self.A)+'[p*'+str(self.ldA)+'+i];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               tensor_temp[i*'+str(self.ldC)+' + j] += weight * '+str(self.B)+'[p*'+str(self.ldB)+' + j];\n'
        if(self.fused_layer):
            if(self.activation_function.name != 'linear'):
                s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+self.fused_layer.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]',self.idx,'i*'+str(self.ldC)+' + j')+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+a+';\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tt(self, m, n, k, A, B):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = k
        self.ldC = n
        a = self.activation_function.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]')

        s = '    // gemm_tt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register sum = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               sum += '+str(self.A)+'[p*'+str(self.ldA)+'+i] * '+str(self.B)+'[j*'+str(self.ldB)+' + p];\n'
        s+= '           }\n'
        s+= '           tensor_temp[i*'+str(self.ldC)+' + j] += sum;\n'
        if(self.fused_layer):
            if(self.activation_function.name != 'linear'):
                s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+self.fused_layer.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]',self.idx,'i*'+str(self.ldC)+' + j')+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] += '+a+';\n'
        s+= '       }\n'
        s+= '    }\n\n'
        
        return s
    
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        if('cst' not in self.previous_layer[0].output_str):
            source_file.write('    for (int k = 0; k < '+str(self.input_channels*self.input_height*self.input_width)+'; ++k){\n        tensor_temp[k] = output_'+str(self.road)+'[k];\n    }\n')
        patch_building_code = self.algo_patch_building_mapping[self.conv_algorithm]()
        source_file.write(patch_building_code)
        source_file.write('    for (int k = 0; k < '+str(self.nb_filters*self.patches_width)+'; ++k){\n        tensor_temp[k] = 0;\n    }\n')

        gemm_code = self.algo_gemm_mapping[self.conv_algorithm](self.nb_filters, self.patches_width, self.patches_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), "output_"+str(self.road))
        source_file.write(gemm_code)
        if(self.data_format == 'channels_last'):
            source_file.write('    for (int f = 0; f < ' + str(self.nb_filters) + '; ++f){\n')
            source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i){\n')
            source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j){\n')
            source_file.write('                output_'+str(self.road)+'[(i * '+str(self.output_width) +' + j) * ' + str(self.nb_filters) + ' + f] = tensor_temp[(f * '+str(self.output_height)+' + i) * '+str(self.output_width)+' + j];\n\n')
            source_file.write('            }\n        }\n    }\n\n')
        else:
            source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = tensor_temp[k];\n    }\n\n')
        
