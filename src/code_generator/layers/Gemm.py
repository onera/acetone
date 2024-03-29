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

#The layer which compute the general matrix multiplication
#input: weight tesnsor W and bias tensor B, input tensor T. The tensor must be of 2D
#data: alpha and beta constante used in the operation, transpo un tuple saying if the tensor T or W must be transposed before the operation
#output: The result of the operation """alpha*T*W + beta*B"""
class Gemm(Layers.Layers):
    def __init__(self, idx, size, alpha, beta, transA, transB, weights, bias, input_shape, output_shape,activation_function):
        super().__init__() 
        self.name = 'Gemm'
        self.idx = idx
        self.size = size
        
        self.alpha = alpha
        self.beta = beta
        self.transpo = (transA,transB)
        self.algo_gemm_mapping = {(0,0):self.write_gemm_nn,
                             (0,1):self.write_gemm_nt,
                             (1,1):self.write_gemm_tt,
                             (1,0):self.write_gemm_tn}
        
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        if(input_shape):
            self.input_height = input_shape[0]
            self.input_width = input_shape[1]
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
        s+= '           float register weight = '+str(self.A)+'[i*'+str(self.ldA)+'+p];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               tensor_temp[i*'+str(self.ldC)+' + j] += weight * '+str(self.B)+'[p*'+str(self.ldB)+' + j];\n'
        s+= '           }\n'
        s+= '       }\n'
        s+= '        for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '            float register output = tensor_temp[i*'+str(self.ldC)+' + j];\n'
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
    #The tensor weight tensor is transposed
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
    #The tensor input tensor is transposed
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
    #Both tensors are transposed
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
    
    def feedforward(self,input):
        input = input.reshape(self.input_height,self.input_width)
        if (self.transpo[0]):
            input = input.transpose()
        if(self.transpo[1]):
            self.weights = self.weights.transpose()
        
        return self.activation_function.compute(self.alpha * input * self.weights + self.beta * self.biases)
    
    def write_to_function_source_file(self,source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int k = 0; k < '+str(self.output_width*self.input_width)+'; ++k){\n        tensor_temp[k] = 0;\n    }\n')
        gemm_code = self.algo_gemm_mapping[self.transpo](self.output_height, self.output_width, self.input_width, self.previous_layer[0].output_str, 'weights_' + self.name + '_' + str("{:02d}".format(self.idx)))
        source_file.write(gemm_code)
        source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = tensor_temp[k];\n    }\n')
        pass

