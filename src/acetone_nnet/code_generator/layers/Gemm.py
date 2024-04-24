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

from ..Layer import Layer
import numpy as np
import pystache

#The layer which compute the general matrix multiplication
#input: weight tesnsor W and bias tensor B, input tensor T. The tensor must be of 2D
#data: alpha and beta constante used in the operation, transpo un tuple saying if the tensor T or W must be transposed before the operation
#output: The result of the operation """alpha*T*W + beta*B"""
class Gemm(Layer):
    def __init__(self, idx, size, alpha, beta, transA, transB, weights, bias, input_shape, output_shape,activation_function):
        super().__init__() 
        self.name = 'Gemm'
        self.idx = idx
        self.size = size
        
        self.alpha = [alpha]
        self.beta =  [beta]
        self.transpo = (transA,transB)
        self.algo_gemm_mapping = {(0,0):self.write_gemm_nn,
                             (0,1):self.write_gemm_nt,
                             (1,1):self.write_gemm_tt,
                             (1,0):self.write_gemm_tn}
        
        self.output_height = output_shape[1]
        self.output_width = output_shape[2]
        if(input_shape):
            if(transA):
                self.input_height = input_shape[2]
                self.input_width = input_shape[1]
            else:
                self.input_height = input_shape[1]
                self.input_width = input_shape[2]
        else:
            self.input_height = 1
            self.input_width = 1
        
        self.weights = np.asarray(weights)
        self.biases = np.asarray(bias)
        self.activation_function = activation_function
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)
        
    #The various ways to compute the operation: 
    
    #None of the tensor ar transposed
    def write_gemm_nn(self, m, n, k, A, B):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['B'] = B
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        mustach_hash['alpha'] = self.alpha
        mustach_hash['beta'] = self.beta
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Gemm/template_gemm_nn.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def write_gemm_nt(self, m, n, k, A, B):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['B'] = B
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        mustach_hash['alpha'] = self.alpha
        mustach_hash['beta'] = self.beta
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Gemm/template_gemm_nt.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tn(self, m, n, k, A, B):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['B'] = B
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('output')
        mustach_hash['alpha'] = self.alpha
        mustach_hash['beta'] = self.beta
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Gemm/template_gemm_tn.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tt(self, m, n, k, A, B):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['m'] = m
        mustach_hash['n'] = n
        mustach_hash['k'] = k
        mustach_hash['A'] = A
        mustach_hash['B'] = B
        mustach_hash['activation_function'] = self.activation_function.write_activation_str('sum')
        mustach_hash['alpha'] = self.alpha
        mustach_hash['beta'] = self.beta
        if (self.fused_layer):
            mustach_hash['fused_layer'] = self.fused_layer.write_activation_str('output_'+str(self.path),self.idx,'i*'+str(self.ldC)+' + j')

            if (self.activation_function.name == 'linear'):
                mustach_hash['linear'] = True

        with open(self.template_path+'layers/Gemm/template_gemm_tt.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)
    
    def forward_path_layer(self,input):
        if(self.transpo[0]):
            input = input.reshape(self.input_width,self.input_height).transpose()
        else:
            input = input.reshape(self.input_height,self.input_width)
            
        if(self.transpo[1]):
            self.weights = self.weights.transpose()
        
        return self.activation_function.compute(self.alpha * np.dot(input,self.weights) + self.beta * self.biases)
    
    def generate_inference_code_layer(self):

        mustach_hash = {}

        mustach_hash['name'] = self.name
        mustach_hash['idx'] = "{:02d}".format(self.idx)
        mustach_hash['comment'] = self.activation_function.comment
        mustach_hash['size'] = self.size
        mustach_hash['road'] = self.path

        mustach_hash['patches_size'] = self.output_width*self.output_height
        mustach_hash['gemm_code'] = self.algo_gemm_mapping[self.transpo](self.output_height, self.output_width, self.input_width, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), self.previous_layer[0].output_str)

        with open(self.template_path+'layers/Gemm/template_Gemm.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)