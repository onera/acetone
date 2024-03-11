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
from abc import ABC, abstractmethod

class Layers(ABC):
    
    def __init__(self):

        self.idx = 0
        self.size = 0
        self.name = ''
        self.next_layer = [] 
        self.previous_layer = []
        self.globalvars_str = ''
        self.header_str = ''
        self.source_str = ''
      
        super().__init__()

    @abstractmethod
    def write_to_layer_c_files(self):
        pass

    @abstractmethod
    def feedforward(self):
        pass

    def flatten_array_orderc(self, array):
    
        flattened_aray = array.flatten(order='C')
        s = '\n        {'
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i])+', '
        s = s[:-2]
        s+='}'
        
        return s
    
    def flatten_array_orderf(self, array):
    
        flattened_aray = array.flatten(order='F')
        s = '\n        {'
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i])+', '
        s = s[:-2]
        s+='}'
        
        return s
    
    def flatten_array_hybrid(self, array):
        ndim = array.ndim
        array = array.reshape(-1, *array.shape[-(ndim-2):])
        
        flattened_aray = array.flatten(order='F')
        s = '\n        {'
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i])+', '
        s = s[:-2]
        s+='}'
        
        return s

    def count_elements_array(self, array):
        nb_elements = 1
        for dim in np.shape(array) : nb_elements *= dim
        return nb_elements

    def compute_padding(self, in_height, in_width, kernel_size, strides, dilation_rate=1):
        
        # Compute 'same' padding tensorflow

        filter_height = (kernel_size - (kernel_size-1)*(dilation_rate-1))
        filter_width = (kernel_size - (kernel_size-1)*(dilation_rate-1))

        # The total padding applied along the height and width is computed as:

        if (in_height % strides == 0):
            pad_along_height = max(filter_height - strides, 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides), 0)
        if (in_width % strides == 0):
            pad_along_width = max(filter_width - strides, 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return pad_right, pad_left, pad_bottom, pad_top

    def create_dicts(self, list_of_dicts, list_of_values_loops):
        
        keys = ['type', 'variable', 'bound', 'start', 'end', 'inner']
        
        function = {}
        function['name'] = self.name
        function['inner'] = []

        for dict, values_loop in zip(list_of_dicts, list_of_values_loops):
            for key, value in zip(keys, values_loop):
                dict[key] = value 

        list_of_dicts.insert(0, function)

        return list_of_dicts
        
    def append_dicts(self, list_of_dicts):
        
        for i in range(0, len(list_of_dicts)-1):
            if 'inner' in list_of_dicts[i].keys():    
                list_of_dicts[i]['inner'].append(list_of_dicts[i+1])
            else:
                list_of_dicts[0]['inner'].append(list_of_dicts[i+1])

        function_dict = list_of_dicts[0]

        return function_dict


class InputLayer(Layers):

    def __init__(self, idx, size):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Input_layer'

    def write_to_function_source_file(self, source_file):
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        output_pre[i] = nn_input[i]; \n    } \n\n')

    def feedforward(self, input):
        
        return input 


class Dense(Layers):

    def __init__(self, idx, size, weights, biases, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Dense'
        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)
        self.activation_function = activation_function
        self.local_var = 'dotproduct'
        
        self.nb_weights = self.count_elements_array(self.weights)
        self.nb_biases = self.count_elements_array(self.biases)

        
    def write_to_function_source_file(self, source_file):
        source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
        source_file.write( '        dotproduct = 0;\n')
        source_file.write( '        for (int j = 0; j < ' + str(self.previous_layer[0].size) + '; ++j)\n        {\n')
        source_file.write( '            dotproduct += output_pre[j] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[(i + ' + str(self.size) + '*j)];\n        }\n')
        source_file.write( '        dotproduct += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[i];\n')

        a = self.activation_function.write_activation_str(self.local_var)

        source_file.write( '        output_cur[i] = '+ a +';\n    }\n\n')

    def feedforward(self, input):

        input = input.reshape((self.previous_layer[0]).size) 

        return self.activation_function.compute((np.dot(input, self.weights) + self.biases))

      
class Conv2D(Layers):
    
    def __init__(self, idx, size, padding, strides, kernel_size, dilation_rate, nb_filters, input_shape, output_shape, weights, biases, activation_function):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Conv2D'
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
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

        if self.padding == 'same':
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.input_height, self.input_width, self.kernel_size, self.strides, self.dilation_rate)
        else:
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = 0, 0, 0, 0


    def write_to_function_source_file(self, source_file):
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    for (int f = 0; f < ' + str(self.nb_filters) + '; ++f)\n    {\n')
        source_file.write('        for (int i = 0; i < '+str(self.output_height)+'; ++i)\n        {\n')
        source_file.write('            for (int j = 0; j < '+str(self.output_width)+'; ++j)\n            {\n')
        source_file.write('                sum = 0;\n')
        source_file.write('                for (int c = 0; c < '+str(self.input_channels)+'; ++c)\n                {\n')
        source_file.write('                    for (int m = 0; m < '+str(self.kernel_size)+'; ++m)\n                    {\n')
        source_file.write('                        for (int n = 0; n < '+str(self.kernel_size)+'; ++n)\n                        {\n')
        source_file.write('                            int ii = i*'+str(self.strides)+' + m*'+str(self.dilation_rate)+' - '+str(self.pad_left)+';\n')
        source_file.write('                            int jj = j*'+str(self.strides)+' + n*'+str(self.dilation_rate)+' - '+str(self.pad_top)+';\n\n')
        source_file.write('                            if (ii >= 0 && ii < '+str(self.input_height)+' && jj >= 0 && jj < '+str(self.input_width)+')\n                            {\n')

        source_file.write('                                sum += output_pre[(ii*'+str(self.input_width)+' + jj)*'+str(self.input_channels)+' + c] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[((m*'+str(self.kernel_size)+' + n)*'+str(self.input_channels)+' + c)*'+str(self.nb_filters)+' + f];\n'  )
        
        source_file.write('                            }\n                        }\n                    }\n                }\n')
        source_file.write('                sum += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[f];\n'            )
        
        a = self.activation_function.write_activation_str(self.local_var)
                    
        source_file.write('                output_cur[(i*'+str(self.output_width)+' + j)*'+str(self.nb_filters)+' + f] = '+ a +';\n')
        source_file.write('            }\n        }\n    }\n\n')
            
    def feedforward(self, input):
        
        input = input.reshape(self.input_height, self.input_width, self.input_channels)
        output = np.zeros((self.output_height, self.output_width, self.nb_filters))
        
        if self.pad_right and self.pad_left and self.pad_top and self.pad_bottom:
            input_padded = np.zeros((self.input_height + self.pad_top + self.pad_bottom, self.input_width + self.pad_left + self.pad_right, self.input_channels))
            input_padded[self.pad_top:-self.pad_bottom, self.pad_left:-self.pad_right, :] = input
        else:
            input_padded = input

        for f in range(self.nb_filters):
            for j in range(self.output_width): 
                for i in range(self.output_height):
                    output[i,j,f]=np.sum(input_padded[i*self.strides:i*self.strides+self.kernel_size, j*self.strides:j*self.strides+self.kernel_size, :] * self.weights[:,:,:,f]) + self.biases[f]
        
        return self.activation_function.compute(output)

class Pooling2D(Layers):
    def __init__(self, idx, size, padding, strides, pool_size, input_shape, output_shape, **kwds):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = ''
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_channels = input_shape[3]
        self.output_height = output_shape[1]
        self.output_width = output_shape[2]
        self.pooling_funtion = ''
        self.local_var = ''
        self.local_var_2 = ''
        self.output_var = ''

        if self.padding == 'same':
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = self.compute_padding(self.input_height, self.input_width, self.pool_size, self.strides)
        else:
            self.pad_right, self.pad_left, self.pad_bottom, self.pad_top = 0, 0, 0, 0

    
    def generate_output_str(self, index, output):
        
        return '    '+output+'['+index+'] = '+ self.output_var +';\n\n'

    @abstractmethod    
    def specific_function(self, index, input_of_layer):
        pass

    def write_to_function_source_file(self, source_file):
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

        source_file.write(self.specific_function('(ii*'+str(self.input_width)+' + jj)*'+str(self.input_channels)+' + c', 'output_pre'))
        source_file.write('                        }\n                    }\n                }\n')
        source_file.write('            ' + self.generate_output_str('(i*'+str(self.output_width)+' + j)*'+str(self.input_channels)+' + c', 'output_cur'))
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
        source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
        source_file.write('    sum = 0;\n\n')
        source_file.write('    for (int i = 0; i < ' + str(self.size) + '; ++i)\n')
        source_file.write('        sum += exp(output_pre[i]);\n\n')
        source_file.write('    for (int j = 0; j < ' + str(self.size) + '; ++j)\n')
        source_file.write('        output_cur[j] = exp(output_pre[j])/sum;\n\n')

    def feedforward(self, input):
        
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp)

        return output
    
   