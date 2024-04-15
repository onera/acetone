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

from .Conv2D_gemm import Conv2D_gemm

import numpy as np
import pystache

class Conv2D_indirect_gemm(Conv2D_gemm):
    """Implements Conv2D using indirect im2col (or im2row) and GeMM"""
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_ppatches(self):
        if (self.pad_right or self.pad_left or self.pad_bottom or self.pad_top):
            self.input_h_padded = self.input_height + self.pad_top + self.pad_bottom
            self.input_w_padded = self.input_width + self.pad_left + self.pad_right

            start_idx = np.arange(self.kernel_h)[:,None]*self.input_w_padded + np.arange(self.kernel_w)
            c=self.input_h_padded*self.input_w_padded*np.arange(self.input_channels)
            start_idx=(c[:,None]+start_idx.ravel()).reshape((-1,self.kernel_h,self.kernel_w))
            offset_idx = np.arange(self.output_height, step=self.strides)[:,None]*self.input_w_padded + np.arange(self.output_width, step=self.strides)
            idx_padded_input = (start_idx.ravel()[:,None] + offset_idx.ravel()).flatten()

            idx_of_zeros = []
            j_zeros = np.concatenate((np.arange(self.pad_left), np.arange(self.pad_right)+(self.input_w_padded-self.pad_right)))
            i_zeros = np.concatenate((np.arange(self.pad_top), np.arange(self.pad_bottom)+(self.input_h_padded-self.pad_bottom)))
            for c in range(self.input_channels):
                for i in range(self.input_h_padded):
                    for j in range(self.input_w_padded):
                        if (np.isin(i, i_zeros) or np.isin(j, j_zeros)):
                            idx_of_zeros.append(j + self.input_w_padded*(i+self.input_h_padded*c))
            
            idx_padded_input = np.where(np.isin(idx_padded_input, idx_of_zeros), np.nan, idx_padded_input)
            _, idx_padded_input = np.unique(idx_padded_input, return_inverse=True) 
            self.ppatches=np.where(idx_padded_input==self.input_shape, np.nan, idx_padded_input)

        else:
            start_idx = np.arange(self.kernel_h)[:,None]*self.input_width + np.arange(self.kernel_w)
            c=self.input_height*self.input_width*np.arange(self.input_channels)
            start_idx=(c[:,None]+start_idx.ravel()).reshape((-1,self.kernel_h,self.kernel_w))
            offset_idx = np.arange(self.output_height, step=self.strides)[:,None]*self.input_width + np.arange(self.output_width, step=self.strides)
            self.ppatches = (start_idx.ravel()[:,None] + offset_idx.ravel()).flatten()
            
        if ('gemm_nt' or 'gemm_tt') in self.conv_algorithm:
            self.ppatches = self.ppatches.reshape((self.patches_height, self.patches_width)).transpose().flatten()  
  
        
        output_str = self.previous_layer[0].output_str
        
        s = '\n        {'
        for i in range(len(self.ppatches)):
            if np.isnan(self.ppatches[i]):
                s += '&zero, '
            else:
                s += '&'+output_str+'[' + str(int(self.ppatches[i])) + '], '

        s=s[:-2]
        s+='}'

        return s

    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['road'] = self.path
        mustach_hash['size'] = self.size

        mustach_hash['activation_function'] = self.activation_function.write_activation_str(self.local_var)

        gemm_code = self.algo_gemm_mapping[self.conv_algorithm](self.nb_filters, self.patches_width, self.patches_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), 'ppatches_' + self.name + '_' + str("{:02d}".format(self.idx)), "output_"+str(self.path), True)
        mustach_hash['gemm_code'] = gemm_code

        if('cst' not in self.previous_layer[0].output_str):
            mustach_hash['cst'] = True
            mustach_hash['prev_size'] = self.input_channels*self.input_height*self.input_width
            
        with open(self.template_path+'layers/Conv/template_Conv_indirect_gemm.c.tpl', 'r') as template_file:
            template = template_file.read()
        template_file.close()        
        
        return pystache.render(template, mustach_hash)
