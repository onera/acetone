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

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class Layers(ABC):
    
    def __init__(self):

        self.idx = 0
        self.size = 0
        self.name = ''
        self.next_layer = [] 
        self.previous_layer = []
        self.road = None
        self.sorted = None
        self.output_str = ''
        self.fused_layer = None
      
        super().__init__()

    @abstractmethod
    def write_to_function_source_file(self):
        pass

    @abstractmethod
    def feedforward(self):
        pass

    def count_elements_array(self, array):
        nb_elements = 1
        for dim in np.shape(array) : nb_elements *= dim
        return nb_elements

    def compute_padding(self,padding, in_height, in_width, kernel_h, kernel_w, strides, dilation_rate=1):
        if(type(padding) == str):
            # Compute 'same' padding tensorflow

            filter_height = (kernel_h - (kernel_h-1)*(dilation_rate-1))
            filter_width = (kernel_w - (kernel_w-1)*(dilation_rate-1))

            # The total padding applied along the height and width is computed as:
            if(padding == 'VALID' or padding == 'valid'):
                pad_right, pad_left, pad_bottom, pad_top = 0,0,0,0
            else:
                if (in_height % strides == 0):
                    pad_along_height = max(filter_height - strides, 0)
                else:
                    pad_along_height = max(filter_height - (in_height % strides), 0)
                if (in_width % strides == 0):
                    pad_along_width = max(filter_width - strides, 0)
                else:
                    pad_along_width = max(filter_width - (in_width % strides), 0)
                
                if((padding == 'SAME_LOWER') or (padding == 'same')):
                    pad_top = pad_along_height // 2
                    pad_bottom = pad_along_height - pad_top
                    pad_left = pad_along_width // 2
                    pad_right = pad_along_width - pad_left
                elif(padding == 'SAME_UPPER'):
                    pad_bottom = pad_along_height // 2
                    pad_top = pad_along_height - pad_bottom
                    pad_right = pad_along_width // 2
                    pad_left = pad_along_width - pad_right        
        else:
            pad_right, pad_left, pad_bottom, pad_top = padding[3], padding[1], padding[2], padding[0]
            
        return pad_right, pad_left, pad_bottom, pad_top
    
    #Give to the layer an string saying were the output will be saved (either in a 'cst' or in an 'output_road')
    def find_output_str(self,dict_cst):
        #dict_cst is the dict linking an layer to the cst in which the must be saved if needed
        
        #either it has to be saved
        if(self in dict_cst):
            output_str = 'cst_'+str(dict_cst[self])
        #Or it can directly go to the next layer
        else:
            output_str = 'output_'+str(self.road)
        self.output_str = output_str
        return self

    def write_activation_and_fusion(self, source_file, output_str, indice, temp_str, space):
        #Allow to have both a fusion (ex: add) and an activation (ex: relu)
        a = self.activation_function.write_activation_str(temp_str)
        #if there is a fusion, we must deal with it
        if(self.fused_layer):
            #if the activation function is anything beside linear, we compute it before the fusion.
            if(self.activation_function.name != 'linear'):
                b=self.fused_layer.write_activation_str(temp_str,self.idx,indice)
                source_file.write(space+temp_str+' = '+a+';\n')
            else:
                b=self.fused_layer.write_activation_str(output_str,self.idx,indice)
            source_file.write(space+output_str+' = '+ b +';\n    }\n\n')
        #if not, we compute the activation function
        else:
            source_file.write(space+output_str+' = '+ a +';\n    }\n\n')

class InputLayer(Layers):

    def __init__(self, idx, size):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Input_layer'

    def write_to_function_source_file(self, source_file):
        
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        output_'+str(self.road)+'[i] = nn_input[i]; \n    } \n\n')

    def feedforward(self, input):
        
        return input 


class Dense(Layers):

    def __init__(self, idx, size, input_shape, weights, biases, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Dense'
        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        self.input_shape = input_shape
        
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        
    def write_to_function_source_file(self, source_file):
        #Variable indicating under which name the input tensor is
        output_str = self.previous_layer[0].output_str
        
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        dotproduct = 0;\n')
        source_file.write( '        for (int j = 0; j < ' + str(self.previous_layer[0].size) + '; ++j)\n        {\n')
        source_file.write( '            dotproduct += '+output_str+'[j] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[(j + ' + str(self.previous_layer[0].size) + '*i)];\n        }\n')
        source_file.write( '        dotproduct += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[i];\n')
        
        self.write_activation_and_fusion(source_file, 'tensor_temp[i]', 'i', self.local_var,'        ')
        source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = tensor_temp[k];\n    }\n\n')

    def feedforward(self, input):
        input = input.reshape(self.previous_layer[0].size)
        return self.activation_function.compute(np.dot(input, self.weights) + self.biases)


class Conv2D(Layers):
    
    def __init__(self, idx, conv_algorithm, data_format, size, padding, strides, kernel_h, kernel_w, dilation_rate, nb_filters, input_shape, output_shape, weights, biases, activation_function):
        super().__init__()
        self.conv_algorithm = conv_algorithm
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = 'Conv2D'
        self.padding = padding
        self.strides = strides
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        
        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]

        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'sum'

        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)
        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.kernel_h, self.kernel_w, self.strides, self.dilation_rate)
    
    @abstractmethod
    def write_to_function_source_file(self, source_file):
        pass

    def feedforward(self, input):
        # Conv for chw
        if(self.data_format == 'channels_last'):
            input = input.reshape(self.input_height, self.input_width, self.input_channels)
            input= np.transpose(input,(2,0,1))
            
        elif(self.data_format == 'channels_first'):
            input = input.reshape(self.input_channels, self.input_height, self.input_width)
        
        output = np.zeros((self.nb_filters, self.output_height, self.output_width))
        print(self.weights.shape)

        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input
            
        for f in range(self.nb_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                        output[f,i,j]=np.sum(input_padded[:, i*self.strides:i*self.strides+self.kernel_h, j*self.strides:j*self.strides+self.kernel_w] 
                                            * self.weights[:,:,:,f]) + self.biases[f]
        if(self.data_format == 'channels_last'):
            output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)

class Conv2D_6loops(Conv2D):
    """Implements Conv2D using the six-loops algorithm (direc conv)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.nb_filters) + '; ++f)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')
        source_file.write('                sum = 0;\n')
        source_file.write('                for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n                {\n')
        source_file.write('                    for (int m = 0; m < '+str(self.kernel_h)+'; ++m)\n                    {\n')
        source_file.write('                        for (int n = 0; n < '+str(self.kernel_w)+'; ++n)\n                        {\n')
        source_file.write('                            int ii = i*'+str(self.strides)+' + m*'+str(self.dilation_rate)+' - '+str(self.pad_left)+';\n')
        source_file.write('                            int jj = j*'+str(self.strides)+' + n*'+str(self.dilation_rate)+' - '+str(self.pad_top)+';\n\n')
        source_file.write('                            if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                            {\n')
        
        source_file.write('                                sum += '+output_str+'[jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[n + '+str(self.kernel_w)+'*(m + '+str(self.kernel_h)+'*(c + '+str(self.input_channels)+'*f))];\n')
        source_file.write('                            }\n                        }\n                    }\n                }\n')                                            
        source_file.write('                sum += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[f];\n'            )
        
        a = self.activation_function.write_activation_str(self.local_var)
        
        if(self.fused_layer):
            b=self.fused_layer.write_activation_str(self.local_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)')
            if(self.activation_function.name != 'linear'):
                source_file.write('                '+self.local_var+' = '+a+';\n')
            source_file.write('                output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)] = '+ b +';\n')
        else:
            source_file.write('                output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*f)] = '+ a +';\n')
        source_file.write('            }\n        }\n    }\n\n')

class Conv2D_gemm(Conv2D):
    """Implements Conv2D using indirect im2col (or im2row) and GeMM"""
   
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

   
    @abstractmethod
    def write_gemm_nn(self, m, n, k, A, B, C):
        pass

    @abstractmethod
    def write_gemm_nt(self, m, n, k, A, B, C):
        pass
    
    @abstractmethod
    def write_gemm_tn(self, m, n, k, A, B, C):
        pass

    @abstractmethod
    def write_gemm_tt(self, m, n, k, A, B, C):
       pass

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
        if('output' in output_str):
            output_str = 'tensor_temp'
        
        s = '\n        {'
        for i in range(len(self.ppatches)):
            if np.isnan(self.ppatches[i]):
                s += '&zero, '
            else:
                s += '&'+output_str+'[' + str(int(self.ppatches[i])) + '], '

        s=s[:-2]
        s+='}'

        return s

    def write_gemm_nn(self, m, n, k, A, B, C, ):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '        for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '            register float weight = '+str(self.A)+'[i*'+str(self.ldA)+'+p];\n'
        s+= '            for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '                tensor_temp[i*'+str(self.ldC)+' + j] += weight * *('+str(self.B)+'[p*'+str(self.ldB)+' + j]);\n'
        s+= '            }\n'
        s+= '        }\n'
        s+= '        for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '            register float output = tensor_temp[i*'+str(self.ldC)+' + j];\n'
        s+= '            output += biases_' + self.name + '_' + str('{:02d}'.format(self.idx))+'[i];\n'

        if(self.fused_layer):
            b=self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')
            if(self.activation_function.name != 'linear'):
                s+= '            output = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+b+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+a+';\n'
        s+= '        }\n'
        s+= '    }\n\n'
        
        return s
    
    def write_gemm_nt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = k
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('output')

        s = '    // gemm_nt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           register float output = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               output += '+str(self.A)+'[i*'+str(self.ldA)+'+p] * *('+str(self.B)+'[j*'+str(self.ldB)+' + p]);\n'
        s+= '           }\n'
        s+= '           output += biases_'+ self.name + '_' + str("{:02d}".format(self.idx))+'[i];\n'

        if(self.fused_layer):
            b=self.fused_layer.write_activation_str('output',self.idx,'i*'+str(self.ldC)+' + j')
            if(self.activation_function.name != 'linear'):
                s+= '            output = '+a+';\n'
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+b+';\n'
        else:
            s+= '            tensor_temp[i*'+str(self.ldC)+' + j] = '+a+';\n'

        s+= '       }\n'
        s+= '    }\n\n'
        
        return s

    def write_gemm_tn(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = n
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]')

        s = '    // gemm_tn\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '           float register weight = '+str(self.A)+'[p*'+str(self.ldA)+'+i];\n'
        s+= '           for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '               tensor_temp[i*'+str(self.ldC)+' + j] += weight * *('+str(self.B)+'[p*'+str(self.ldB)+' + j]);\n'
        
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

    def write_gemm_tt(self, m, n, k, A, B, C):

        self.m = m
        self.n = n
        self.k = k
        self.A = A
        self.ldA = m
        self.B = B
        self.ldB = k
        self.C = C
        self.ldC = n
        a = self.activation_function.write_activation_str('tensor_temp[i*'+str(self.ldC)+' + j]')

        s = '    // gemm_tt\n'
        s+= '    for (int i=0; i<'+str(self.m)+'; i++){\n'
        s+= '       for(int j=0; j<'+str(self.n)+'; ++j){\n'
        s+= '           float register sum = 0;\n'
        s+= '           for (int p=0; p<'+str(self.k)+'; ++p){\n'
        s+= '               sum += '+str(self.A)+'[p*'+str(self.ldA)+'+i] * *('+str(self.B)+'[j*'+str(self.ldB)+' + p]);\n'
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
            self.source_file.write('                output_'+str(self.road)+'[(i * '+str(self.output_width) +' + j) * ' + str(self.nb_filters) + ' + f] = tensor_temp[(f * '+str(self.output_height)+' + i) * '+str(self.output_width)+' + j];\n\n')
            self.source_file.write('            }\n        }\n    }\n\n')
        else:
            source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = tensor_temp[k];\n    }\n\n')
        
        return 0

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
        s+= '                    output_'+str(self.road)+'[i*'+str(self.patches_width)+' + j] = '+output_str+'[(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj];\n'
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
        s+= '                    output_'+str(self.road)+'[j*'+str(self.patches_height)+' + i] = '+output_str+'[(c_offset*'+str(self.input_height)+' + ii)*'+str(self.input_width)+' + jj];\n'
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
        
    def feedforward(self, input):
        # Conv for chw
        if(self.data_format == 'channels_last'):
            input = input.reshape(self.input_height, self.input_width, self.input_channels)
            input= np.transpose(input,(2,0,1))
            
        elif(self.data_format == 'channels_first'):
            input = input.reshape(self.input_channels, self.input_height, self.input_width)
        
        output = np.zeros((self.nb_filters, self.output_height, self.output_width))
        print(self.weights.shape)

        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_channels, self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right))
            input_padded[:, self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right] = input
        else:
            input_padded = input
            
        for f in range(self.nb_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                        output[f,i,j]=np.sum(input_padded[:, i*self.strides:i*self.strides+self.kernel_h, j*self.strides:j*self.strides+self.kernel_w] 
                                            * self.weights[:,:,:,f]) + self.biases[f]
        
        if(self.data_format == 'channels_last'):
            output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)

class Pooling2D(Layers):
    def __init__(self, idx, data_format, size, padding, strides, pool_size, input_shape, output_shape,**kwargs):
        
        super().__init__()
        self.idx = idx
        self.data_format = data_format
        self.size = size
        self.name = ''
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size

        if self.data_format == 'channels_first':
            self.input_channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.output_height = output_shape[2]
            self.output_width = output_shape[3]

        elif self.data_format == 'channels_last':
            self.input_height = input_shape[1]
            self.input_width = input_shape[2]
            self.input_channels = input_shape[3]
            self.output_height = output_shape[1]
            self.output_width = output_shape[2]

        self.pooling_funtion = ''
        self.local_var = ''
        self.local_var_2 = ''
        self.output_var = ''

        self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.padding,self.input_height, self.input_width, self.pool_size,self.pool_size, self.strides)

    @abstractmethod    
    def specific_function(self, index, input_of_layer):
        pass

    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')

        source_file.write('            ' + self.update_local_vars())

        source_file.write('                for (int m = 0; m < '+str(self.pool_size)+'; ++m)\n                {\n')
        source_file.write('                    for (int n = 0; n < '+str(self.pool_size)+'; ++n)\n                    {\n')
        source_file.write('                        int ii = i*'+str(self.strides)+' + m - '+str(self.pad_left)+';\n')
        source_file.write('                        int jj = j*'+str(self.strides)+' + n - '+str(self.pad_top)+';\n\n')
        source_file.write('                        if (ii >= 0 && ii < '+str( self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                        {\n')
        source_file.write(self.specific_function('jj + '+str(self.input_width)+'*(ii + '+str(self.input_height)+'*c)', output_str))
        source_file.write('                        }\n                    }\n                }\n')
        
        if (self.fused_layer):
            b = self.fused_layer.write_activation_str(self.output_var,self.idx,'j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)')
        else: 
            b = self.output_var

        source_file.write('            output_'+str(self.road)+'[j + '+str(self.output_width)+'*(i + '+str(self.output_height)+'*c)]'+' = '+ b +';\n')
        
        source_file.write('            }\n        }\n    }\n\n')

    def feedforward(self, input):

        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        output = np.zeros((self.output_height, self.output_width, self.input_channels))
        
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right, self.input_channels))
            input_padded[self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right, :] = input
        else:
            input_padded = input

        for c in range(self.input_channels):
            for j in range(self.output_width): 
                for i in range(self.output_height):
                    output[i,j,c]= self.pooling_function((input_padded[i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size, c]))
        return output

class AveragePooling2D(Pooling2D):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        
        self.name = 'AveragePooling2D'
        self.pooling_function = np.mean
        self.local_var = 'sum'
        self.local_var_2 = 'count'
        self.output_var = self.local_var + '/' + self.local_var_2

    def declare_local_vars(self, data_type):
        
        s = '    '+ data_type + ' '+ self.local_var +';\n'
        s += '    int '+ self.local_var_2 + ';\n\n'

        return s

    def update_local_vars(self):

        s = '    '+ self.local_var + ' = 0; '+ self.local_var_2 + ' = 0;\n'
  
        return s

    def specific_function(self, index, input_of_layer):
        # Computes the average in this subclass AveragePooling2D 
        s = '                            '+self.local_var+' += '+input_of_layer+'['+index+'];\n'
        s += '                            '+self.local_var_2+' ++;\n'
        
        return s

class MaxPooling2D(Pooling2D):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        
        self.name = 'MaxPooling2D'
        self.pooling_function = np.amax
        self.local_var = 'max'
        self.output_var = self.local_var

    def declare_local_vars(self, data_type):
        
        s = '    '+ data_type + ' '+ self.local_var +';\n\n'

        return s

    def update_local_vars(self):
        
        s = '    '+ self.local_var +' = -INFINITY;\n'

        return s

    def specific_function(self, index, input_of_layer):
        s = '                            if ('+input_of_layer+'['+index+'] > '+self.local_var+')\n'
        s += '                                '+self.local_var+' = '+input_of_layer+'['+index+'];\n'
    
        return s

class Softmax(Layers):

    def __init__(self, idx, size):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'

    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    sum = 0;\n\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; ++i)\n')
        source_file.write('        sum += exp('+output_str+'[i]);\n\n')
        source_file.write('    for (int j = 0; j < ' + str(self.size) + '; ++j)\n    {\n')
        source_file.write('        output_'+str(self.road)+'[j] = exp('+output_str+'[j])/sum;\n\n')
        if (self.fused_layer):
            b = self.fused_layer.write_activation_str('output_'+str(self.road)+'[j]', self.idx, 'j')
            source_file.write('        output_'+str(self.road)+'[j] = '+ b +';\n')
        source_file.write('    }\n\n')

    def feedforward(self, input):
        
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp)

        return output
    
#Concatenate two tensor alongside a given axis
#attribut: axis alongside of which the concatenation will be done
#input: a list of tensor to concatenate
#output: the concatenated tensor 
class Concatenate(Layers):
    def __init__(self, idx, size, axis, input_shapes,output_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = 'Concatenate'
        self.axis = axis
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]
        self.activation_function = activation_function

    def write_concat(self, source_file):
        borne_sup = 0
        borne_inf = 0
        for k in range(len(self.previous_layer)):
            input_shape = self.input_shapes[k]
            output_str = self.previous_layer[k].output_str
            #We take the value of the matrix only if the indices are inside the adequat limits
            #the max indice is always at one length of matrix after the min indice
            
            if (self.axis == 1):
                #concat alongside the channels
                borne_sup += input_shape[1]
                source_file.write('                if((f < '+str(borne_sup) +') && (f >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[j  + ' + str(input_shape[3]) + ' * (i + ' + str(input_shape[2]) + ' * (f - ' + str(borne_inf) + ') )];\n                }\n')
                borne_inf += input_shape[1]
            if (self.axis == 2):
                #concat alongside the height
                borne_sup += input_shape[2]
                source_file.write('                if((i < '+str(borne_sup) +') && (i >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[j  + ' + str(input_shape[3]) + ' * ( (i - ' + str(borne_inf) + ') + ' + str(input_shape[2]) + ' * f )];\n                }\n')
                borne_inf += input_shape[2]
            if (self.axis == 3):
                #concat alongside the width
                borne_sup += input_shape[3]
                source_file.write('                if((j < '+str(borne_sup) +') && (j >= '+str(borne_inf) +'))\n                {\n')
                source_file.write('                    tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
                source_file.write(output_str+'[(j - ' + str(borne_inf) + ') + ' + str(input_shape[3]) + ' * ( i + ' + str(input_shape[2]) + ' * f )];\n                }\n')
                borne_inf += input_shape[3]
            

            
    
    def write_to_function_source_file(self, source_file):
        #we go through all the indices of the tensor then write the opeartion depending on the axis
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.write_concat(source_file)
        source_file.write('            }\n        }\n    }\n\n')
        a = self.activation_function.write_activation_str('tensor_temp[k]')
        source_file.write('    for (int k = 0; k < '+str(self.size)+'; ++k){\n        output_'+str(self.road)+'[k] = '+a+';\n    }\n')

    def feedforward(self, inputs):
        output = inputs[0]
        output = output.reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        for i in range(1,len(inputs)):
            input = inputs[i]
            input = input.reshape(self.input_shapes[i][1],self.input_shapes[i][2],self.input_shapes[i][3])
            output = np.concatenate((output, input),axis=self.axis - 1) 
        return self.activation_function.compute(output)

#Do the dotproduct of two tensors
#atribut: the axis alongside of which the dot product will be done. if type(axis) == int, same axis for the two tensor. can be tuples of axis (element i of axis represent axis of tensor i)
#input: two tensor
#output: the resultant tensor 
class Dot(Layers):
    def __init__(self, idx, size, axis, input_shapes,output_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.input_shapes = input_shapes
        self.name = 'Dot'
        #we seek a tuple of axis
        if (type(axis) == int):
            self.axis = [axis,axis]
        else:
            self.axis = axis
        self.output_fourth_dim = output_shape[1]
        self.output_height = output_shape[3]
        self.output_width = output_shape[4]
        self.output_channels = output_shape[2]
        self.activation_function = activation_function
    
    def write_dot(self,source_file, i):
        #depend if it's the first or the second tensor on the dot 
        if (i == 0):
            var = ['f','g']
            size = [self.output_channels,self.output_fourth_dim]
        else:
            var = ['j','i']
            size = [self.output_width,self.output_height]
        
        output_str = self.previous_layer[i].output_str
        source_file.write(output_str + '[')
        #the k indice correspond to the interation in the dot product, and it's position depend of the axis of the product.
        if (self.axis[i] == 2):
            source_file.write('k + '+ str(self.input_shapes[i][self.axis[i]]) + ' * ('+ var[0] +' + ' + str(size[0]) + ' * ' + var[1] + ')]')
        elif(self.axis[i] == 1):
            source_file.write(var[0] + ' + '+ str(size[0]) + ' * (k + ' + str(self.input_shapes[i][self.axis[i]]) + ' * ' + var[1] + ')]')
        elif(self.axis[i] == 0):
            source_file.write(var[0] + ' + '+ str(size[0]) + ' * ('+ var[1] +' + ' + str(size[1]) + ' * k)]')

            

    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for(int g = 0; g < ' + str(self.output_fourth_dim) + '; g++)\n    {\n')
        source_file.write('        for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n        {\n')#the two non variant dim of the first tensor
        source_file.write('            for (int i = 0; i < ' + str(self.output_height) + '; i++)\n            {\n')
        source_file.write('                for (int j = 0; j < ' + str(self.output_width) + '; j++)\n                {\n')#the two non variant dim of the second tensor
        source_file.write('                    register float output = 0;\n')
        source_file.write('                    for (int k = 0; k < ' + str(self.input_shapes[0][self.axis[0]]) + '; k++)\n                    {\n')
        source_file.write('                        output +=')
        self.write_dot(source_file,0)
        source_file.write(' * ')
        self.write_dot(source_file,1)
        #Apply the activation function and/or the fused function to the output of the layer
        a = self.activation_function.write_activation_str('output')
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * (f + ' + str(self.output_channels) +' * g))] = '+ a +';\n')
        source_file.write(';\n')
        source_file.write('                    }\n                }\n            }\n        }\n    }\n\n')
    
    def feedforward(self, inputs):
        inputs[0] = inputs[0].reshape(self.input_shapes[0][1],self.input_shapes[0][2],self.input_shapes[0][3])
        inputs[1] = inputs[1].reshape(self.input_shapes[1][1],self.input_shapes[1][2],self.input_shapes[1][3])
        return self.activation_function.compute(np.tensordot(inputs[0],inputs[1],axes=[self.axis[0]-1,self.axis[1]-1]))

#The layer which compute the general matrix multiplication
#input: weight tesnsor W and bias tensor B, input tensor T. The tensor must be of 2D
#data: alpha and beta constante used in the operation, transpo un tuple saying if the tensor T or W must be transposed before the operation
#output: The result of the operation """alpha*T*W + beta*B"""
class Gemm(Layers):
    def __init__(self, idx, size, alpha, beta, transA, transB, weight, bias, input_shape, output_shape,activation_function):
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
        
        self.weights = np.asarray(weight)
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


#The class of the Layers which compute operation with broadcast numpy style
#attribut: none
#input: a list of tensor
#output: the resultant tensor
class Broadcast(Layers):
    def __init__(self, idx, size, input_shapes, output_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ''
        self.input_shapes = input_shapes
        
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        self.output_channels = output_shape[1]

        self.activation_function = activation_function
    
    @abstractmethod
    def specific_operator(self,source_file):
        pass

    def write_add_a_tensor(self,source_file): #a function to do the broadcast numpy style
        #Min, Max and Avg follow the rule of broadcasting but need a operator before the iteration of previous layer 
        if(self.name == 'Maximum'):
            source_file.write('max(')
        elif(self.name == 'Minimum'):
            source_file.write('min(')
        elif(self.name == 'Average'):
            source_file.write('(')
        #Going through all the ancestors 
        for k in range(len(self.previous_layer)):
            if(k!=0):
                self.specific_operator(source_file) #the operator to broadcats
            input_shape = self.input_shapes[k]
            output_str = self.previous_layer[k].output_str #if the value is in a road or saved eslewhere
            #The % operator allow to always saty in teh range of the tensor's indices 
            source_file.write(output_str+'[(j % '+ str(input_shape[3]) + ') + ' + str(input_shape[3]) + ' * ((i % ' + str(input_shape[2]) + ') + ' + str(input_shape[2]) + ' * (f % '+str(input_shape[1])+'))]' )
        
        #Closing the operator of Max,Min and Avg
        if(self.name == 'Maximum' or self.name=='Minimum'):
            source_file.write(')')
        elif(self.name=='Average'):
            source_file.write(')/('+ str(len(self.previous_layer)) +')\n')
        
        source_file.write(';\n')
    
    #Go through all the indices and do the operation
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n             {\n')
        source_file.write('                tensor_temp[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
        self.write_add_a_tensor(source_file)
        source_file.write('            }\n        }\n    }\n\n ')
        a = self.activation_function.write_activation_str('tensor_temp[f]')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {        output_'+str(self.road)+'[f] = '+a+';\n    }\n')
    
    @abstractmethod
    def feedforward(self, inputs):
        pass

#Addition of several tensor
class Add(Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Add'
    
    def specific_operator(self,source_file):
        return source_file.write(' + ')
    
    def feedforward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output += input
        return self.activation_function.compute(output)

#Multiplication of several tensors
class Multiply(Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Multiply'
    
    def specific_operator(self,source_file):
        return source_file.write(' * ')
    
    def feedforward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output *= input
        return self.activation_function.compute(output)

#Subtraction of several tensors
class Subtract(Broadcast):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Subtract'
    
    def specific_operator(self,source_file):
        return source_file.write(' - ')
    
    def feedforward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output -= input
        return self.activation_function.compute(output)

#Division of several tensors
class Divide(Broadcast):
    
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

#Return a tensor with where each position (f,i,j) contains the max of all the values at position (f,i,j) in each tensor
class Maximum(Broadcast):
    def __init__(self, idx, size):
        super().__init__(idx, size)
        self.name = 'Maximum'
    
    def specific_operator(self, source_file):
        source_file.write(', ')

    def feedforward(self, inputs):
        maxi = inputs[0]
        for input in inputs[1:]:
            maxi = np.maximum(maxi, input)
        return self.activation_function.compute(maxi)

#Return a tensor with where each position (f,i,j) contains the min of all the values at position (f,i,j) in each tensor
class Minimum(Broadcast):
    def __init__(self, idx, size):
        super().__init__(idx, size)
        self.name = 'Minimum'
    
    def specific_operator(self, source_file):
        source_file.write(', ')

    def feedforward(self, inputs):
        mini = inputs[0]
        for input in inputs[1:]:
            mini = np.minimum(mini, input)
        return self.activation_function.compute(mini)

#Return a tensor with where each position (f,i,j) contains the average of all the values at position (f,i,j) in each tensor
class Average(Broadcast):
    def __init__(self, idx, size):
        super().__init__(idx, size)
        self.name = 'Average'

    def specific_operator(self, source_file):
        source_file.write(' + ')

    def feedforward(self, inputs):
        output = inputs[0]
        for input in inputs[1:]:
            output +=input 
        return self.activation_function.compute(output/(len(inputs)))
    
#The class of the Layers which compute element wise operation
#attribut: none
#input: a tensor 
#output: the resultant tensor
############################ EQUIVALENT TO ACTIVATION FUNCTIONS
class Element_Wise(Layers):
    def __init__(self, idx, size):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ''
    
    @abstractmethod
    def specific_operator(self,source_file): #The operation to apply
        pass    
    
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; i++)\n    {\n') #going through all the elements without regard of their position
        source_file.write('        output_'+str(self.road)+'[i] = ')
        self.specific_operator(source_file) #to operaiton to do element wise
        source_file.write('    }\n\n')
    
    @abstractmethod
    def feedforward(self, input):
        pass


class Exponential(Element_Wise):
    def __init__(self, idx, size):
        super().__init__(idx, size)
        self.name = 'Exponential'
    
    def specific_operator(self, source_file):
        output_str = self.previous_layer[0].output_str
        return source_file.write('exp('+output_str+'[i]);')
    
    def feedforward(self, input):
        return np.exp(input)
    
class Logarithm(Element_Wise):
    def __init__(self, idx, size):
        super().__init__(idx, size)
        self.name = 'Logarithm'
    
    def specific_operator(self, source_file):
        output_str = self.previous_layer[0].output_str
        return source_file.write('log('+output_str+'[i]);')
    
    def feedforward(self, input):
        return np.log(input)

class ReLu(Element_Wise):
    
    def __init__(self,idx,size):
        super().__init__(idx,size)
        self.name = 'relu'
    
    def feedforward(self, input):
        return np.maximum(0,input)
    
    def specific_operator(self, source_file):
        output_str = self.previous_layer[0].output_str

        return source_file.write(output_str+'[i]  > 0 ? '+output_str+'[i] : 0;\n') # output = condition ? value_if_true : value_if_false
        
#Confine each element of the entry between two values
#attribut: a value min and a values max which define the intervalle of observation
#input: a tensor
#output: the resultant tensor
class Clip(Layers):
    def __init__(self, idx, size, max, min):
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Clip'
        self.max = max
        self.min = min
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; i++)\n    {\n')#going through all the elements of the tensor
        source_file.write('        if('+output_str+'[i] < '+ str(self.min) +')\n        {\n')#if it's smaller than the min, it takes the min value
        source_file.write('            output_'+str(self.road)+'[i] = ' + str(self.min) + ';\n        }else{\n')
        source_file.write('            if('+output_str+'[i] > '+ str(self.max) +')\n            {\n')#if it's bigger than the max, it takes the max value
        source_file.write('                output_'+str(self.road)+'[i] = ' + str(self.max) + ';\n            }else{\n')
        source_file.write('                    output_'+str(self.road)+'[i] = '+output_str+'[i];\n            }\n')
        source_file.write('        }\n    }\n\n')

    def feedforward(self, input):
        return np.clip(input, self.min,self.max)
    
#extract a list of subtensor from a given tensor
#attribut: axis alongside of which the submatrix will be extracted (if the desired submatrix must have the height, width or channels of the parent tensor)
#input: a tensor
#output: a list of tensor
class Gather(Layers):
    
    def __init__(self, idx, size, axis,  indices, input_shape, output_shape):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Gather'
        self.indices = indices
        self.axis = axis
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]
        
        
    def write_loops(self,source_file):
        flat_indices = np.ndarray.flatten(np.array(self.indices)) #may have indices like: [[0,1],[1,2]]
        source_file.write('    int indice['+str(len(flat_indices))+'] = {') #the list of the indices to use
        for i in flat_indices:
            source_file.write(str(i))
            if(i != flat_indices[-1]):
                source_file.write(", ")
            else:
                source_file.write("};\n")
        #the indice that will change depend on the axis. 
        if(self.axis == 1): #collecting channels
            source_file.write('    for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n    {\n')
            source_file.write('        int f = indice[k];\n')
            source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
            source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        if(self.axis == 2): #collecting heights
            source_file.write('    for (int f = 0; f < ' + str(self.input_channels) + '; f++)\n    {\n')
            source_file.write('        for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n        {\n')
            source_file.write('            int i = indice[k];\n')
            source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        if(self.axis == 3): #collecting widths
            source_file.write('    for (int f = 0; f < ' + str(self.input_channels) + '; f++)\n    {\n')
            source_file.write('        for (int i = 0; i < ' + str(self.input_height) + '; i++)\n        {\n')
            source_file.write('            for (int k = 0; k < ' + str(len(self.indices)) + '; k++)\n            {\n')
            source_file.write('                int j = indice[k];\n')
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    int position = 0;\n')#to know the position of the next element to add
        self.write_loops(source_file)
        source_file.write('                output_cur_'+str(self.road)+'[position]'+
                          ' = '+output_str+'[j + '+str(self.input_width)+' * (i + '+str(self.input_height)+' * f)];\n')
        source_file.write('                position++;\n')
        source_file.write('            }\n        }\n    }\n\n ')
        
    def feedforward(self,input):
        input = input.reshape(self.input_channels,self.input_height,self.input_width)
        return np.take(input, indices=self.indices, axis=self.axis-1)
    
#The resize Layers
#Only Height and Width are resized
#attribut: a lot of stuff
#input: a tensor to be resized, the desired size or a scale to multiply the size by, the region of interest
#output:the resized tensor
##############  https://onnx.ai/onnx/operators/onnx__Resize.html for more informations
#The strategie is always to go throught the elements, find the coordinate in the original tensor 
# and apply a transformation to the value in the original tensor to find the value to enter in the new tensor
class Resize(Layers):
    
    def __init__(self,idx,size,input_shape,axes,coordinate_transformation_mode,exclude_outside,
                 keep_aspect_ratio_policy,boolean_resize,target_size,roi,extrapolation_value, 
                 nearest_mode,activation_function):
        super().__init__()
        self.idx = idx
        if(type(nearest_mode) == bytes):
            self.nearest_mode = str(nearest_mode)[2:-1]
        else: 
            self.nearest_mode = nearest_mode
        
        self.activation_function = activation_function
        self.size = size
        self.axes = axes
        self.exclude_outside = exclude_outside
        self.keep_aspect_ratio_policy = keep_aspect_ratio_policy
        self.roi = roi
        self.name = 'Resize'
        self.extrapolation_value = extrapolation_value
        if(type(coordinate_transformation_mode) == bytes):
            self.coordinate_transformation_mode = str(coordinate_transformation_mode)[2:-1]
        else: 
            self.coordinate_transformation_mode = coordinate_transformation_mode
        self.coordinate_transformation_mode_mapping = {"half_pixel":self.half_pixel, 
                                                       "half_pixel_symmetric":self.half_pixel_symmetric,
                                                       "pytorch_half_pixel":self.pytorch_half_pixel,
                                                       "align_corners":self.align_corners,
                                                       "asymmetric":self.asymmetric,
                                                       "tf_crop_and_resize":self.tf_crop_and_resize}
        #if channel first
        self.input_channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        if (boolean_resize):
            self.scale = target_size
            self.output_channels = int(self.input_channels*target_size[1])
            self.output_height = int(self.input_height*target_size[2])
            self.output_width = int(self.input_width*target_size[3])
        else:
            self.output_channels = target_size[1]
            self.output_height = target_size[2]
            self.output_width = target_size[3]
            self.scale[1] = self.output_channels / self.input_channels
            self.scale[2] = self.output_height / self.input_height
            self.scale[3] = self.output_width / self.input_width
            
    @abstractmethod
    def feedforward(self,input):
        pass
    
    @abstractmethod
    def write_to_function_source_file(self, source_file):
        pass

    #Defining the several coordinate transformations. cf documentation 
    def half_pixel(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += coord_original + ' = ('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        return s
    
    def half_pixel_symmetric(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += 'float adjustment = ' + str(int(self.output_width)) + ' / ' + str(self.output_width)  +';\n'
        s += '                float center = ' + str(self.input_width) + ' / 2;\n'
        s += '                float offset = center * (1 - adjustment);\n'
        s +='                '+ coord_original + ' = offset + ('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        return s
    
    def pytorch_half_pixel(self,coord_resized,coord_dim,coord_original):
        s = '                '
        if(coord_dim==2):
            length = self.output_height
        else: length = self.output_width
        s += coord_original + ' = '
        if (length > 1):
            s += '('+ coord_resized+' + 0.5) / '+ str(self.scale[coord_dim])+' - 0.5;\n'
        else:
            s += '0;\n' 
        return s
    
    def align_corners(self,coord_resized,coord_dim,coord_original):
        s = '                '
        if(coord_dim==2):
            length_original = self.input_height
            length_resized = self.output_height
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            
        s += coord_original + ' = ' +coord_resized+' * (' + str(length_original)+' - 1) / (' + str(length_resized)+' - 1);\n'
        return s
    
    def asymmetric(self,coord_resized,coord_dim,coord_original):
        s = '                '
        s += coord_original + ' = '+ coord_resized+' / '+ str(self.scale[coord_dim]) +';\n'
        return s
    
    def tf_crop_and_resize(self,coord_resized,coord_dim,coord_original):
        if(coord_dim==2):
            length_original = self.input_height
            length_resized = self.output_height
            start = self.roi[2]
            end = self.roi[6]
        else: 
            length_original = self.input_width
            length_resized = self.output_width
            start = self.roi[3]
            end = self.roi[7]
        
        s = '                '
        s += coord_original + ' = ' 
        if(length_resized > 1):
            s+= str(start) + ' * ('+ str(length_original)+' - 1) + '+ coord_resized+' * (' +str(end)+' - '+str(start)+') * ('+str(length_original)+' - 1) / (' + str(length_resized)+' - 1);\n'
        else:
            s+= '0.5 * (' +str(end)+' - '+str(start)+') * ('+str(length_original)+' - 1);\n'
        s+= '                if(('+coord_original+' < 0) || ('+coord_original+' > '+str(length_original)+'){'+coord_original+' = '+ str(self.extrapolation_value)+'}\n'
        return s
    
#The mode Nearest of the Resize layers.
#The value in the new tensor is found by applying an rounding operation
class ResizeNearest(Resize):
    
    def __init__(self, idx, size,input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size,input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                        scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode='nearest'
        self.nearest_mode_mapping = {"round_prefer_floor":self.round_prefer_floor,
                                     "round_prefer_ceil":self.round_prefer_ceil,
                                     "floor":self.floor,
                                     "ceil":self.ceil}
    
    #Defining the several method to chose the nearest
    def floor(self,x):
        return '                '+str(x)+' = floor('+str(x)+');\n'
    
    def ceil(self,x,y):
        return '                '+str(x)+' = ceil('+str(y)+');\n'
    
    def round_prefer_floor(self,x,y):
        return '                '+str(x)+' = floor(ceil(2 * ' + str(y) + ') / 2);\n'
    
    def round_prefer_ceil(self,x,y):
        return '                '+str(x)+' = ceil(floor(2 * ' + str(y) + ') / 2);\n'
    
    def write_to_function_source_file(self, source_file):
        output_str = self.previous_layer[0].output_str
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
        source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
        source_file.write(self.nearest_mode_mapping[self.nearest_mode]('x'))#Choosing the closest coordinate in the original tensor
        source_file.write(self.nearest_mode_mapping[self.nearest_mode]('y'))
        a = self.activation_function.write_activation_str(output_str+'[ y + ' + str(self.input_width) + ' * (x + ' + str(self.input_height) + ' * f)]')
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        
        source_file.write('            }\n        }\n    }\n\n')
    
    
    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = (tf.image.resize(input, [self.output_height,self.output_width], method='nearest')).numpy() #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)
    
#The mode Linear of the Resize layers
#The value in the output tensor are found thanks to a (bi)linear interpolation
class ResizeLinear(Resize):
    
    def __init__(self, idx, size, input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size, input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                         scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode = 'linear'
        
    def bilinear_interpolation(self):
        #the equation for the bilinear interpolation
        s = '(f11 * (x2 - x) * (y2 - y) +'
        s+= ' f21 * (x - x1) * (y2 - y) +'
        s+= ' f12 * (x2 - x) * (y - y1) +'
        s+= ' f22 * (x - x1) * (y - y1))' 
        s+= ' / ((x2 - x1) * (y2 - y1));\n'
        return s
    
    def write_cst_interpolation_2D(self):
        #the four points for a 2D interpolation and their values
        s = '    y2 = 0;\n'
        s+= '    y1 = '+str(self.input_height-1)+';\n'
        s+= '    x2 = '+str(self.input_width-1)+';\n'
        s+= '    x1 = 0;\n'
        return s
    
    #a function to write the point used in the interpolation
    def write_function_values_interpolation_2D(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f12 = '+output_str+'[y1 + ' + str(self.input_width) + ' * (x2 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f22 = '+output_str+'[y2 + ' + str(self.input_width) + ' * (x2 + ' + str(self.input_height) + ' * f)];\n'
        s+= '    f21 = '+output_str+'[y2 + ' + str(self.input_width) + ' * (x1 + ' + str(self.input_height) + ' * f)];\n'
        return s        
    
    #The function to do the interpolation but in 1D
    def write_cst_interpolation_1D_width(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the width of the tensor
        s = '    x2 = '+str(self.input_width-1)+';\n'
        s+= '    x1 = 0;\n'
        return s
    
    def write_function_values_interpolation_1D_width(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x1 + ' + str(self.input_width) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x2 + ' + str(self.input_width) + ' * f];\n'
        return s
    
    def write_cst_interpolation_1D_height(self):
        #the two points for a 1D interpolation and their values if the non void dimension is the height of the tensor
        s = '    x2 = '+str(self.input_height-1)+';\n'
        s+= '    x1 = 0;\n'
        return s

    def write_function_values_interpolation_1D_height(self):
        output_str = self.previous_layer[0].output_str
        s = '    f11 = '+output_str+'[x1 + ' + str(self.inGemmput_height) + ' * f];\n'
        s+= '    f22 = '+output_str+'[x2 + ' + str(self.input_height) + ' * f];\n'
        return s
    
    def linear_interpolation(self):
        #the equation for the interpolation
        s = '(f11 * (x2 - x) +'
        s+= ' f22 * (x - x1))' 
        s+= ' / (x2 - x1);\n'
        return s
    
    #To differentiate the interpolation needed: prevent division by 0
    def write_cst_interpolation(self):
        if ((self.input_height == 1) and (self.input_width > 1)):
            return self.write_cst_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            return self.write_cst_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
            return self.write_cst_interpolation_2D()
        
    def write_function_values_interpolation(self):
        if ((self.input_height == 1) and (self.input_width > 1)):
            return self.write_function_values_interpolation_1D_width()
        elif((self.input_height > 1) and (self.input_width == 1)):
            return self.write_function_values_interpolation_1D_height()
        elif((self.input_height > 1) and (self.input_width > 1)):
            return self.write_function_values_interpolation_2D()

    def interpolation(self):
        if ((self.input_height > 1) and (self.input_width >1)):
            return self.bilinear_interpolation()
        else:
            return self.linear_interpolation()
    
    def transforme_coordinate(self,source_file):
        if ((self.input_height == 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x'))
        elif((self.input_height > 1) and (self.input_width == 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))
        elif((self.input_height > 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
    
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write(self.write_cst_interpolation()) #The point used in the interpolation
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write(self.write_function_values_inteGemmrpolation()) #f in the value of the allement f_i_i
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.transforme_coordinate(source_file) #Finding the coordinate in the original tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = ')
        source_file.write(self.interpolation()) #Doing the interpolation to find the output value
        if(self.activation_function.name != 'linear'):
            a = self.activation_function.write_activation_str('output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)]')
            source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+ a)
        source_file.write('            }\n        }\n    }\n\n')

    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bilinear').numpy() #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)

#The Cubic mode of the Resize layers
#Use a (bi)cubic interpolation to find the new value
class ResizeCubic(Resize):
    
    def __init__(self, idx, size, input_shape, axes=[], coordinate_transformation_mode='half_pixel', exclude_outside=0, 
                 keep_aspect_ratio_policy='stretch', scale=[], target_size=[], roi=[], extrapolation_value=0,
                 cubic_coeff_a = -0.75,nearest_mode = 'round_prefer_floor'):
        super().__init__(idx, size, input_shape, axes, coordinate_transformation_mode, exclude_outside, keep_aspect_ratio_policy, 
                         scale, target_size, roi, extrapolation_value,nearest_mode)
        self.mode = 'cubic'
        self.cubic_coeff_a = cubic_coeff_a
    
    def feedforward(self, input):
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        input= np.transpose(input,(2,0,1))#Function resize in tensorflow take a format channel last
        output = tf.image.resize(input, [self.output_height,self.output_width], method='bicubic') #No numpy method for this layer
        output= np.transpose(output,(1,2,0))
        return self.activation_function.compute(output)
    
    #Compute the simple cubic convolution as describe in https://ieeexplore.ieee.org/document/1163711 (cf doc ONNX)
    def cubic_convolution_interpolation(self,f_1,f0,f1,f2,x):
        #give the values to the constants of the interpolation
        s = '                f_1 = ' + f_1 + ';\n'
        s+= '                f0 = ' + f0 + ';\n'
        s+= '                f1 = ' + f1 + ';\n'
        s+= '                f2 = ' + f2 + ';\n'
        s+= '                s = ' + x + '-floor('+x+');\n'
        s+= '                result_interpolation = '
        #the value of the variable of interest: the result of the interpolation
        s+= 'f_1 * a * s * (1 + s * (s - 2)) + '
        s+= 'f0 * (s * s *(a * (s - 1) + 2 * s - 3) + 1) + '
        s+= 'f1 * s * (s * (-s * (2 + a) + 2 * a + 3) - a) + '
        s+= 'f2 * a * s * s * (1 - s);\n'
        return s
    
    #To do the bicubic interpolation, you need to do 4 cubic interpolation alongside a dimension,
    #Then you use this result to do a cubic interpolation alongside the last dimension
    #cf https://en.wikipedia.org/wiki/Bicubic_interpolation
    def bicubic_convolution_interpolation(self,f_1_1,f0_1,f1_1,f2_1,f_10,f00,f10,f20,f_11,f01,f11,f21,f_12,f02,f12,f22):
        s = self.cubic_convolution_interpolation(f_1_1,f0_1,f1_1,f2_1,'x','b_1')
        s+= self.cubic_convolution_interpolation(f_10,f00,f10,f20,'x','b0')
        s+= self.cubic_convolution_interpolation(f_11,f01,f11,f21,'x','b1')
        s+= self.cubic_convolution_interpolation(f_12,f02,f12,f22,'x','b2')
        s+= self.cubic_convolution_interpolation('b_1','b0','b1','b2','y','result_interpolation')
        return s
    
    def transforme_coordinate(self,source_file):
        if ((self.input_height == 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'x'))
        elif((self.input_height > 1) and (self.input_width == 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))
        elif((self.input_height > 1) and (self.input_width > 1)):
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('i',2,'x'))#Finding the coordinate in the original tensor
            source_file.write(self.coordinate_transformation_mode_mapping[self.coordinate_transformation_mode]('j',3,'y'))
    
    #The function writing the interpoaltion(s) in C
    def interpolation(self):
        output_str = self.previous_layer[0].output_str
        #Setting up the values
        if ((self.input_height > 1) and (self.input_width >1)):
            #All the values used to calculate the output
            f_1_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f0_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f1_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f2_1 = output_str+'[y0-1 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_10 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f00 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f10 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f20 = output_str+'[y0 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_11 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f01 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f11 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f21 = output_str+'[y0+1 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            f_12 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0-1 + ' + str(self.input_height) + ' * f)];\n'
            f02 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0 + ' + str(self.input_height) + ' * f)];\n'
            f12 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0+1 + ' + str(self.input_height) + ' * f)];\n'
            f22 = output_str+'[y0+2 + ' + str(self.input_width) + ' * (x0+2 + ' + str(self.input_height) + ' * f)];\n'
            return self.bicubic_convolution_interpolation(f_1_1,f0_1,f1_1,f2_1,f_10,f00,f10,f20,f_11,f01,f11,f21,f_12,f02,f12,f22)
        elif((self.input_height > 1) and (self.input_width == 1)):
            f_1 = output_str+'[x0-1 + ' + str(self.input_height) + ' * f];\n'
            f0 = output_str+'[x0 + ' + str(self.input_height) + ' * f];\n'
            f1 = output_str+'[x0+1 + ' + str(self.input_height) + ' * f];\n'
            f2 = output_str+'[x0+2 + ' + str(self.input_height) + ' * f];\n'
            return self.cubic_convolution_interpolation(f_1,f0,f1,f2,'x','result_interpolation')
        elif((self.input_height == 1) and (self.input_width > 1)):
            f_1 = output_str+'[x0-1 + ' + str(self.input_width) + ' * f];\n'
            f0 = output_str+'[x0 + ' + str(self.input_width) + ' * f];\n'
            f1 = output_str+'[x0+1 + ' + str(self.input_width) + ' * f];\n'
            f2 = output_str+'[x0+2 + ' + str(self.input_width) + ' * f];\n'
            return self.cubic_convolution_interpolation(f_1,f0,f1,f2,'x','result_interpolation')
            
    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    a = '+str(self.cubic_coeff_a)+';\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.transforme_coordinate(source_file) #Finding the coordinate in the original tensor
        source_file.write('                int x0 = floor(x);\n                int y0 = floor(y);\n')
        source_file.write(self.interpolation())#getting the output
        a = self.activation_function.write_activation_str('result_interpolation')
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        source_file.write('            }\n        }\n    }\n\n')
        

#The Pad Layers
#Pad alongside each dimmensions
#attribut: the mode of padding required
#input: a tensor to be padded, the desired pads, the value of teh constant if mode == constant
#output:the resized tensor
######################### cf https://onnx.ai/onnx/operators/onnx__Pad.html for the doc
class Pad(Layers):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__()
        self.idx = idx
        self.size = size
        self.pads = pads
        self.constant_value = constant_value
        self.axes = axes
        self.name = 'Pad'
        self.input_shape = input_shape
        self.output_channels = input_shape[1] + pads[1] + pads[5]
        self.output_height = input_shape[2] + pads[2] + pads[6]
        self.output_width = input_shape[3] + pads[3] + pads[7]
        self.mode = ''
        self.activation_function = activation_function
    
    def feedforward(self, input):
        input = input.reshape(self.input_shape[1], self.input_shape[2], self.input_shape[3])
        nb_dim = len(self.pads)//2
        pad_width = [(self.pads[i],self.pads[i+nb_dim]) for i in range(1,nb_dim)] #Constructing the pads accordingly to the numpy nomenclature
        return self.activation_function.compute(np.pad(input,pad_width=pad_width,mode=self.mode,constant_values=self.constant_value,))
    
    @abstractmethod
    def write_padding(self,source_file):
        pass
    
    def write_to_function_source_file(self, source_file,):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.output_channels) + '; f++)\n    {\n')#going through all the elements of the resized tensor
        source_file.write('        for (int i = 0; i < ' + str(self.output_height) + '; i++)\n        {\n')
        source_file.write('            for (int j = 0; j < ' + str(self.output_width) + '; j++)\n            {\n')
        self.write_padding(source_file) #Depend on how the padding is done
        source_file.write('            }\n        }\n    }\n\n')

#The Constant mode of the Pad layers
#Use a constant to fill paddings
class Constant_Pad(Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes,input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes,input_shape,activation_function)
        self.mode = 'constant'
    
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[(j -'+str(self.pads[3])+') + ' + str(self.input_shape[3]) + ' * ((i -'+str(self.pads[2])+') + ' + str(self.input_shape[2]) + ' * f)]')
        b = self.activation_function.write_activation_str(str(self.constant_value))
        #if the new indice is in the the old tensor, we take the value from the tensor

        source_file.write('                if((f >= '+str(self.pads[1])+') && (f <'+str(self.output_channels - self.pads[5])+') && (i >= '+str(self.pads[2])+') && (i <'+str(self.output_height - self.pads[6])+') && (j >= '+str(self.pads[3])+') && (j <'+str(self.output_width - self.pads[7])+'))\n')
        source_file.write('                {\n')
        source_file.write('                    output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
        source_file.write('                }\n                else\n')#else, we take the constant value
        source_file.write('                {\n')
        source_file.write('                    output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+b+';\n')
        source_file.write('                }\n')
    
#The Reflect mode of the Pad layers
#Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
class Reflect_pad(Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes, input_shape,activation_function)
        self.mode = 'reflect'
    
    #A function to find the correponding coordinates in the input tensor
    def write_reflect(self,new_indice,indice_max,old_indice,pad):
        s = '                '+new_indice+' = ('+str(pad)+' - '+old_indice+') % (2 * '+str(indice_max)+');\n'
        s+= '                if('+new_indice+' > '+str(indice_max)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = 2 * '+str(indice_max)+' - '+new_indice+';\n'
        s+= '                }\n'
        return s
        
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write(self.write_reflect('new_f',self.input_shape[1]-1,'f',self.pads[1]))#Finding the correponding coordinates
        source_file.write(self.write_reflect('new_i',self.input_shape[2]-1,'i',self.pads[2]))
        source_file.write(self.write_reflect('new_j',self.input_shape[3]-1,'j',self.pads[3]))
        #Giving the value to the output tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')

#The Edge mode of the Pad layers
#Pads with the edge values of array.
class Edge_pad(Pad):
    
    def __init__(self, idx, size, pads, constant_value, axes, input_shape,activation_function):
        super().__init__(idx, size, pads, constant_value, axes, input_shape,activation_function)
        self.mode = 'edge'
    
    #Either go the the edge coordinates, or stay with the same coordinates
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write('                new_f =  f - '+self.pads[1]+';\n')#take the indice in the input tensor
        source_file.write('                new_i =  i - '+self.pads[2]+';\n')
        source_file.write('                new_j =  j - '+self.pads[3]+';\n')
        source_file.write(self.write_change_indice('f',self.pads[1],self.input_shape[1],'new_f'))#Going to an edge if necessary
        source_file.write(self.write_change_indice('i',self.pads[2],self.input_shape[2],'new_i'))
        source_file.write(self.write_change_indice('j',self.pads[3],self.input_shape[3],'new_j'))
        #Giving the value to the output tensor
        source_file.write('                output_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')

#The Wrap mode of the Pad layers
#Pads with the wrap of the vector along the axis. 
#The first values are used to pad the end and the end values are used to pad the beginning.
class Wrap_pad(Pad):
    
    def __init__(self, idx, size, mode, pads, constant_value, axes, input_shape):
        super().__init__(idx, size, mode, pads, constant_value, axes, input_shape)
        self.mode = 'wrap'
    
    #Translate to the value used to wrap the input tensor
    def write_change_indice(self,old_indice,pad,dim,new_indice):
        s = '                if(('+old_indice+' < '+str(pad)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = ('+new_indice+') % ('+str(dim)+');\n'
        s+= '                }\n'
        s+= '                if(('+old_indice+' >= '+str(pad + dim)+')\n'
        s+= '                {\n'
        s+= '                    '+new_indice+' = ('+new_indice+') % ('+str(dim)+');\n'
        s+= '                }\n'
        return s
        
    def write_padding(self,source_file):
        output_str = self.previous_layer[0].output_str
        a = self.activation_function.write_activation_str(output_str+'[new_j + ' + str(self.input_shape[3]) + ' * (new_i + ' + str(self.input_shape[2]) + ' * new_f)]')

        source_file.write('                new_f =  f - '+self.pads[1]+';\n')#take the indice in the input tensor
        source_file.write('                new_i =  i - '+self.pads[2]+';\n')
        source_file.write('                new_j =  f - '+self.pads[3]+';\n')
        source_file.write(self.write_change_indice('f',self.pads[1],self.input_shape[1],'new_f'))#If necessary, taking the wrapping value
        source_file.write(self.write_change_indice('i',self.pads[2],self.input_shape[2],'new_i'))
        source_file.write(self.write_change_indice('j',self.pads[3],self.input_shape[3],'new_j'))
        #Giving the value to the output tensor
        source_file.write('                output_cur_'+str(self.road)+'[j + ' + str(self.output_width) + ' * (i + ' + str(self.output_height) + ' * f)] = '+a+';\n')
