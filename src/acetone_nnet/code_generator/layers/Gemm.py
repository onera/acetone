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
from ..activation_functions import ActivationFunctions
import numpy as np
import pystache

#The layer which compute the general matrix multiplication
#input: weight tesnsor W and bias tensor B, input tensor T. The tensor must be of 2D
#data: alpha and beta constante used in the operation, transpo un tuple saying if the tensor T or W must be transposed before the operation
#output: The result of the operation """alpha*T*W + beta*B"""
class Gemm(Layer):
    def __init__(self, idx:int, size:int, alpha:float|int, beta:float|int, transA:bool|int, transB:bool|int, weights:np.ndarray, bias:np.ndarray, input_shape:list, output_shape:list, activation_function:ActivationFunctions):
        super().__init__() 
        self.name = 'Gemm'
        self.idx = idx
        self.size = size
        
        if alpha != 1:
            self.alpha = [alpha]
        else:
            self.alpha = []
        if beta != 1:
            self.beta =  [beta]
        else:
            self.beta = []
            
        self.transpo = (transA,transB)
        self.algo_gemm_mapping = {(0,0):self.write_gemm_nn,
                             (0,1):self.write_gemm_nt,
                             (1,1):self.write_gemm_tt,
                             (1,0):self.write_gemm_tn}
        
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        if(input_shape):
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
        else:
            self.input_height = 1
            self.input_width = 1
        
        self.weights = weights
        self.biases = bias
        self.activation_function = activation_function
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        ####### Checking the instantiation#######

        ### Checking argument type ###
        if  type(self.idx)!= int:
            raise TypeError("Error: idx type in Gemm (idx must be int)")
        if  type(self.size)!= int:
            raise TypeError("Error: size type in Gemm (size must be int)")
        if  type(self.alpha[0]) != float and type(self.alpha[0]) == int:
            raise TypeError("Error: alpha type in Gemm (alpha must be int or float)")
        if  type(self.beta[0]) != float and type(self.beta[0]) == int:
            raise TypeError("Error: beta type in Gemm (beta must be int or float)")
        if any(type(self.transpo[i]) != int and type(self.transpo[i]) != bool for i in range(2)):
            raise TypeError("Error: transpose type in Gemm (must be boolean or int)")
        if type(self.output_height) != int:
            raise TypeError("Error: output height type in Gemm (must be int)")
        if type(self.output_width) != int:
            raise TypeError("Error: output width type in Gemm (must be int)")
        if type(self.input_height) != int:
            raise TypeError("Error: input height type in Gemm (must be int)")
        if type(self.input_width) != int:
            raise TypeError("Error: input width type in Gemm (must be int)")
        if type(self.weights) != np.ndarray:
            raise TypeError("Error: weights in Gemm (weights must be an numpy array)")
        if type(self.biases) != np.ndarray:
            raise TypeError("Error: biases in Gemm (biases must be an numpy array)")
        if not isinstance(self.activation_function, ActivationFunctions):
            raise TypeError("Error: activation function type in Gemm (activation function must be a sub-classe of acetone_nnet Activation Function)")

        ### Checking value consistency ###
        if self.size != self.output_height*self.output_width:
            raise ValueError("Error: size value in Gemm ("+str(self.size)+"!="+str(self.output_height*self.output_width)+")")
        shape = self.input_height if self.transpo[0] else self.input_width
        if self.weights.shape[self.transpo[1]] != shape:
            raise ValueError("Error: non consistency between weight shape and input shape in Gemm ("+str(self.weights.shape[self.transpo[1]])+"!="+str(shape)+")")
        shape = self.input_width if self.transpo[0] else self.input_height
        if self.output_height != shape:
            raise ValueError("Error: non consistency between input shape and output shape in Gemm ("+str(self.output_height)+"!="+str(shape)+")")
        if self.output_width != self.weights.shape[1-self.transpo[1]]:
            raise ValueError("Error: non consistency between output shape and weight shape in Gemm ("+str(self.output_width)+"!="+str(self.weights.shape[1-self.transpo[1]])+")")
        if any(self.biases.shape[i] != 1 and self.biases.shape[i] != output_shape[3-i] for i in range(len(self.biases.shape))):
            raise ValueError("Error: biases in Gemm not broadcastable to dim "+str((self.output_height,self.output_width)))

        
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
    
    def forward_path_layer(self, input:np.ndarray):
        input = input.reshape(self.input_height,self.input_width)
        if(self.transpo[0]):
            input = input.transpose()
            
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
        if self.transpo[0]:
            mustach_hash['gemm_code'] = self.algo_gemm_mapping[self.transpo](self.output_height, self.output_width, self.input_height, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), self.previous_layer[0].output_str)
        else:
            mustach_hash['gemm_code'] = self.algo_gemm_mapping[self.transpo](self.output_height, self.output_width, self.input_width, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)), self.previous_layer[0].output_str)

        with open(self.template_path+'layers/Gemm/template_Gemm.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)