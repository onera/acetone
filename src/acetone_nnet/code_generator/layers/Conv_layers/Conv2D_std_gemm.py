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

import pystache

class Conv2D_std_gemm(Conv2D_gemm):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.algo_patch_building_mapping = { 'gemm_nn' : self.write_im2col,
                                             'gemm_nt' : self.write_im2row,
                                             'gemm_tn' : self.write_im2col,
                                             'gemm_tt' : self.write_im2row}

    def write_im2col(self):
        output_str = self.previous_layer[0].output_str
        if('output' in output_str):
            output_str = 'tensor_temp'
        
        mustach_hash = {}

        mustach_hash['patches_height'] = self.patches_height
        mustach_hash['kernel_w'] = self.kernel_w
        mustach_hash['kernel_h'] = self.kernel_h
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['strides'] = self.strides
        mustach_hash['pad_top'] = self.pad_top
        mustach_hash['pad_left'] = self.pad_left
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width
        mustach_hash['road'] = self.path
        mustach_hash['patches_width'] = self.patches_width
        mustach_hash['output_str'] = output_str

        with open(self.template_path+'layers/Conv/template_im2col.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def write_im2row(self):

        output_str = self.previous_layer[0].output_str
        if('output' in output_str):
            output_str = 'tensor_temp'
        
        mustach_hash = {}

        mustach_hash['patches_height'] = self.patches_height
        mustach_hash['kernel_w'] = self.kernel_w
        mustach_hash['kernel_h'] = self.kernel_h
        mustach_hash['output_height'] = self.output_height
        mustach_hash['output_width'] = self.output_width
        mustach_hash['strides'] = self.strides
        mustach_hash['pad_top'] = self.pad_top
        mustach_hash['pad_left'] = self.pad_left
        mustach_hash['input_height'] = self.input_height
        mustach_hash['input_width'] = self.input_width
        mustach_hash['road'] = self.path
        mustach_hash['patches_width'] = self.patches_width
        mustach_hash['output_str'] = output_str

        with open(self.template_path+'layers/Conv/template_im2row.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['size'] = self.size
        mustach_hash['road'] = self.path

        mustach_hash['patch_building_code'] = self.algo_patch_building_mapping[self.conv_algorithm]()
        mustach_hash['patches_size'] = self.nb_filters*self.patches_width
        mustach_hash['gemm_code'] = self.algo_gemm_mapping[self.conv_algorithm](self.nb_filters, self.patches_width, self.patches_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), "output_"+str(self.path),'tensor_temp',False)

        if('cst' not in self.previous_layer[0].output_str):
            mustach_hash['cst'] = True
            mustach_hash['input_size'] = self.input_channels*self.input_height*self.input_width
        

        with open(self.template_path+'layers/Conv/template_Conv_std_gemm.c.tpl','r') as template_file:
            tempalte = template_file.read()
        template_file.close()

        return pystache.render(tempalte, mustach_hash)
