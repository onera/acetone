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

    def generate_flowfacts_dict(self):

        if self.version == 'v1_small_nets' or self.version == 'v1_large_nets' or self.version == 'v1_2':
            
            keys = ['type', 'variable', 'bound', 'start', 'end', 'inner']
            
            function = {}
            function['name'] = self.name
            function['inner'] = []

            # Create dictionary for each loop present in function
            for dict, values_loop in zip(self.list_of_dicts, self.list_of_values_loops):
                for key, value in zip(keys, values_loop):
                    dict[key] = value 

            self.list_of_dicts.insert(0, function)

            for i in range(0, len(self.list_of_dicts)-1):
                if 'inner' in self.list_of_dicts[i].keys():    
                    self.list_of_dicts[i]['inner'].append(self.list_of_dicts[i+1])
                else:
                    self.list_of_dicts[0]['inner'].append(self.list_of_dicts[i+1])

            function_dict = self.list_of_dicts[0]

            return function_dict

class InputLayer(Layers):

    def __init__(self, idx, size):
       
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Input_layer'

    def write_to_layer_c_files(self, data_type, version, layers_source_file, layers_header_file):

        if version == 'v1':
            layers_source_file.write('int Input_layer(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output) \n{ \n')
            layers_source_file.write('    for (int i = 0; i < net[layer_idx].layer_size; ++i) \n    { \n')
            layers_source_file.write('        output[i] = input[i]; \n    } \n\n')
            layers_source_file.write('    return 0; \n} \n\n')
            
            layers_header_file.write('int Input_layer(int layer_idx, ' + data_type + ' *input, ' + data_type + ' *output);\n')

    def write_to_function_source_file(self, data_type, version, source_file):
        
        if version == 'v2':
            source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
            source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
            source_file.write( '        output_pre[i] = nn_input[i]; \n    } \n\n')

        elif version == 'v3':
            source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n\n')

        else:
            pass

    def write_to_function_header_file(self, version, header_file):
        
        if version == 'v1':
            header_file.write('#define l0_size ' + str(self.size) + '\n\n')

        return    

    def write_to_globalvars_file(self, version, data_type, globalvars_file):
        
        if version == 'v1':
            
            globalvars_file.write('    [0] = {\n')
            globalvars_file.write('        .layer_type = Input_layer,\n')
            globalvars_file.write('        .layer_size = l'+str(self.idx)+'_size,\n')
            globalvars_file.write('        .pad_right = 0x0,\n')
            globalvars_file.write('        .pad_left = 0x0,\n')
            globalvars_file.write('        .pad_bottom = 0x0,\n')
            globalvars_file.write('        .pad_top = 0x0,\n')
            globalvars_file.write('        .strides = 0x0,\n')
            globalvars_file.write('        .pool_size = 0x0,\n')
            globalvars_file.write('        .kernel_size = 0x0,\n')
            globalvars_file.write('        .dilation_rate = 0x0,\n')
            globalvars_file.write('        .nb_filters = 0x0,\n')
            globalvars_file.write('        .input_height = 0x0,\n')
            globalvars_file.write('        .input_width = 0x0,\n')
            globalvars_file.write('        .input_channels = 0x0,\n')
            globalvars_file.write('        .output_height = 0x0,\n')
            globalvars_file.write('        .output_width = 0x0,\n')
            globalvars_file.write('        .weights = 0x0,\n')
            globalvars_file.write('        .biases = 0x0,\n')
            globalvars_file.write('        .actv_function = 0x0\n')
            globalvars_file.write('        },\n')

        return 

    def feedforward(self, input):
        
        return input 

    def generate_flowfacts_dict(self, version):

        self.version = version

        if self.version == 'v1':
            
            values_loop_layer_size = ['for loop', 'i', ('l'+str(self.idx)+'_size'), 0, self.size - 1]  
            self.list_of_values_loops = [values_loop_layer_size]

            loop_layer_size = {}
            self.list_of_dicts = [loop_layer_size]

            return Layers.generate_flowfacts_dict(self)

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

    def write_to_layer_c_files(self, data_type, version, layers_source_file, layers_header_file):
        
        if version == 'v1':


            layers_source_file.write('int Dense(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output) \n{ \n')
            layers_source_file.write('    '+ data_type + ' dotproduct;\n\n')
            layers_source_file.write('    for (int i = 0; i < net[layer_idx].layer_size; ++i) \n    { \n')
            layers_source_file.write('        dotproduct = 0;\n')
            layers_source_file.write('        for (int j = 0; j < net[layer_idx-1].layer_size; ++j)\n        {\n')
            layers_source_file.write('            dotproduct += input[j] * (net[layer_idx].weights[(j*net[layer_idx].layer_size+i)]);\n        }\n')
            layers_source_file.write('        dotproduct += net[layer_idx].biases[i];\n')
            layers_source_file.write('        output[i] = net[layer_idx].actv_function(dotproduct);\n    }\n\n')
            layers_source_file.write('    return 0; \n} \n\n')
            
            layers_header_file.write('int Dense(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output);\n')
        
    def write_to_function_source_file(self, data_type, version, source_file):

        if version == 'v2':
            source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
            source_file.write( '    for (int i = 0; i < ' + str(self.size) + '; ++i) \n    { \n')
            source_file.write( '        dotproduct = 0;\n')
            source_file.write( '        for (int j = 0; j < ' + str(self.previous_layer[0].size) + '; ++j)\n        {\n')
            source_file.write( '            dotproduct += output_pre[j] * weights_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[(i + ' + str(self.size) + '*j)];\n        }\n')
            source_file.write( '        dotproduct += biases_' + self.name + '_' + str("{:02d}".format(self.idx)) + '[i];\n')

            a = self.activation_function.write_activation_str(self.local_var)

            source_file.write( '        output_cur[i] = '+ a +';\n    }\n\n')

        elif version == 'v3':
            if self.idx == 1: input_of_layer = 'nn_input'
            else: input_of_layer = 'output_pre'

            source_file.write(  '    // ' + self.name + '_' + str(self.idx) + '\n')
            for i in range(self.size):
                source_file.write( '    dotproduct = 0;\n')
                for j in range(self.previous_layer[0].size):
                    source_file.write( '    dotproduct += ' +input_of_layer+ '['+str(j)+'] * '+ str(self.weights.flatten()[i+self.size*j]) +';\n')
                source_file.write( '    dotproduct += '+ str(self.biases.flatten()[i]) +';\n')
                
                a = self.activation_function.write_activation_str(self.local_var)

                source_file.write( '    output_cur['+str(i)+'] = '+ a +';\n\n')
            
        else:
            pass  

    def write_to_function_header_file(self, version, header_file):
        
        if version == 'v1':
            header_file.write('#define l'+str(self.idx)+'_size ' + str(self.size) + '\n\n')

        else:
            pass

    def write_to_globalvars_file(self, version, data_type, globalvars_file):
      
        if version == 'v1':

            globalvars_file.write('    ['+str(self.idx)+'] = {\n' )
            globalvars_file.write('        .layer_type = Dense,\n')
            globalvars_file.write('        .layer_size = l'+str(self.idx)+'_size,\n')
            globalvars_file.write('        .pad_right = 0x0,\n')
            globalvars_file.write('        .pad_left = 0x0,\n')
            globalvars_file.write('        .pad_bottom = 0x0,\n')
            globalvars_file.write('        .pad_top = 0x0,\n')
            globalvars_file.write('        .strides = 0x0,\n')
            globalvars_file.write('        .pool_size = 0x0,\n')
            globalvars_file.write('        .kernel_size = 0x0,\n')
            globalvars_file.write('        .dilation_rate = 0x0,\n')
            globalvars_file.write('        .nb_filters = 0x0,\n')
            globalvars_file.write('        .input_height = 0x0,\n')
            globalvars_file.write('        .input_width = 0x0,\n')
            globalvars_file.write('        .input_channels = 0x0,\n')
            globalvars_file.write('        .output_height = 0x0,\n')
            globalvars_file.write('        .output_width = 0x0,\n')
            globalvars_file.write('        .weights = weights_'+ self.name + '_' + str("{:02d}".format(self.idx)) + ',\n')
            globalvars_file.write('        .biases = biases_'+ self.name + '_' + str("{:02d}".format(self.idx)) + ',\n')
            globalvars_file.write('        .actv_function =  '+ (self.activation_function).name +',\n        },\n')
  
    def feedforward(self, input):

        input = input.reshape((self.previous_layer[0]).size) 

        return self.activation_function.compute((np.dot(input, self.weights) + self.biases))

    def generate_flowfacts_dict(self, version):
        
        self.version = version

        if self.version == 'v1':
            
        
            values_loop_layer_size = ['for loop', 'i', ('l'+str(self.idx)+'_size'), 0, (self.size - 1), []]
            values_loop_previous_layer_size = ['for loop', 'j', ('l'+str(self.idx - 1)+'_size'), 0, (self.previous_layer[0].size - 1)]

            self.list_of_values_loops = [values_loop_layer_size, values_loop_previous_layer_size]

            loop_layer_size = {}
            loop_previous_layer_size = {}
            
            self.list_of_dicts = [loop_layer_size, loop_previous_layer_size]

            return Layers.generate_flowfacts_dict(self)
      
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

    def write_to_layer_c_files(self, data_type, version, layers_source_file, layers_header_file):

        if version == 'v1':
            
            layers_source_file.write('int Conv2D(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output) \n{ \n')
            layers_source_file.write('    '+ data_type + ' sum;\n\n')
            layers_source_file.write('    for (int f = 0; f < net[layer_idx].nb_filters; ++f)\n    {\n')
            layers_source_file.write('        for (int i = 0; i < net[layer_idx].output_height; ++i)\n        {\n')
            layers_source_file.write('            for (int j = 0; j < net[layer_idx].output_width; ++j)\n            {\n')
            layers_source_file.write('                sum = 0;\n')
            layers_source_file.write('                for (int c = 0; c < net[layer_idx].input_channels; ++c)\n                {\n')
            layers_source_file.write('                    for (int m = 0; m < net[layer_idx].kernel_size; ++m)\n                    {\n')
            layers_source_file.write('                        for (int n = 0; n < net[layer_idx].kernel_size; ++n)\n                        {\n')
            layers_source_file.write('                            int ii = i*net[layer_idx].strides + m*net[layer_idx].dilation_rate - net[layer_idx].pad_left;\n')
            layers_source_file.write('                            int jj = j*net[layer_idx].strides + n*net[layer_idx].dilation_rate - net[layer_idx].pad_top;\n\n')
            layers_source_file.write('                            if (ii >= 0 && ii < net[layer_idx].input_height && jj >= 0 && jj < net[layer_idx].input_width)\n                            {\n')
            layers_source_file.write('                                sum += input[(ii*net[layer_idx].input_width + jj)*net[layer_idx].input_channels + c] * net[layer_idx].weights[((m*net[layer_idx].kernel_size + n)*net[layer_idx].input_channels + c)*net[layer_idx].nb_filters + f];\n'  )
            layers_source_file.write('                            }\n                        }\n                    }\n                }\n')
            layers_source_file.write('                sum += net[layer_idx].biases[f];\n'            )
            layers_source_file.write('                output[(i*net[layer_idx].output_width + j)*net[layer_idx].nb_filters + f] = net[layer_idx].actv_function(sum);\n')
            layers_source_file.write('            }\n        }\n    }\n\n    return 0;\n}\n\n')

            layers_header_file.write('int Conv2D(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output);\n')

    def write_to_function_source_file(self, data_type, version, source_file):
         
        if version == 'v2':
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
            
        elif version == 'v3':
            
            if self.idx == 1: input_of_layer = 'nn_input'
            else: input_of_layer = 'output_pre'

            source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
            for f in range(self.nb_filters):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        source_file.write('    sum = 0;\n' )
                        s = '' # string to store sum
                        for c in range(self.input_channels):
                            for m in range(self.kernel_size):
                                for n in range(self.kernel_size):
                                    ii = i*self.strides + m*self.dilation_rate - self.pad_left
                                    jj = j*self.strides + n*self.dilation_rate - self.pad_top                                
                                    if ii >= 0 and ii <self.input_height and jj >= 0 and jj < self.input_width :
                                        
                                        s += '    sum += ' + input_of_layer + '['+str((ii*self.input_width + jj)*self.input_channels+c)+'] * '+ str((self.weights.flatten())[((m*self.kernel_size + n)*self.input_channels+c)*self.nb_filters + f]) +';\n'

                                    else:
                                        continue

                        source_file.write(s)
                        source_file.write('    sum += '+ str((self.biases.flatten())[f]) +';\n'            )
                        a = self.activation_function.write_activation_str(self.local_var)

                        source_file.write('    output_cur['+str((i*self.output_width+ j)*self.nb_filters+ f)+'] = '+ a +';\n\n')

        else:
            pass    

    def write_to_function_header_file(self, version, header_file):
    
        if version == 'v1':
            
            header_file.write('#define l'+str(self.idx)+'_size ' + str(self.size) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_right ' + str(self.pad_right) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_left ' + str(self.pad_left) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_bottom ' + str(self.pad_bottom) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_top ' + str(self.pad_top) + '\n')
            header_file.write('#define l'+str(self.idx)+'_strides ' + str(self.strides) + '\n')
            header_file.write('#define l'+str(self.idx)+'_kernel_size ' + str(self.kernel_size) + '\n')
            header_file.write('#define l'+str(self.idx)+'_dilation_rate ' + str(self.dilation_rate) + '\n')
            header_file.write('#define l'+str(self.idx)+'_nb_filters ' + str(self.nb_filters) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_height ' + str(self.input_height) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_width ' + str(self.input_width) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_channels ' + str(self.input_channels) + '\n')
            header_file.write('#define l'+str(self.idx)+'_output_height ' + str(self.output_height) + '\n')
            header_file.write('#define l'+str(self.idx)+'_output_width ' + str(self.output_width) + '\n\n')

        else:
            pass

    def write_to_globalvars_file(self, version, data_type, globalvars_file):
        
        if version == 'v1':
            globalvars_file.write('    ['+str(self.idx)+'] = {\n' )
            globalvars_file.write('        .layer_type = Conv2D,\n')
            globalvars_file.write('        .layer_size = l'+str(self.idx)+'_size,\n'   )
            globalvars_file.write('        .pad_right = l'+str(self.idx)+'_pad_right,\n')
            globalvars_file.write('        .pad_left = l'+str(self.idx)+'_pad_left,\n')
            globalvars_file.write('        .pad_bottom = l'+str(self.idx)+'_pad_bottom,\n')
            globalvars_file.write('        .pad_top = l'+str(self.idx)+'_pad_top,\n')
            globalvars_file.write('        .strides = l'+str(self.idx)+'_strides,\n')
            globalvars_file.write('        .pool_size = 0x0,\n')
            globalvars_file.write('        .kernel_size = l'+str(self.idx)+'_kernel_size,\n')
            globalvars_file.write('        .dilation_rate = l'+str(self.idx)+'_dilation_rate,\n')
            globalvars_file.write('        .nb_filters = l'+str(self.idx)+'_nb_filters,\n')
            globalvars_file.write('        .input_height = l'+str(self.idx)+'_input_height,\n')
            globalvars_file.write('        .input_width = l'+str(self.idx)+'_input_width,\n')
            globalvars_file.write('        .input_channels = l'+str(self.idx)+'_input_channels,\n')
            globalvars_file.write('        .output_height = l'+str(self.idx)+'_output_height,\n')
            globalvars_file.write('        .output_width = l'+str(self.idx)+'_output_width,\n')
            globalvars_file.write('        .weights = weights_'+ self.name + '_' + str("{:02d}".format(self.idx)) + ',\n')
            globalvars_file.write('        .biases = biases_'+ self.name + '_' + str("{:02d}".format(self.idx)) + ',\n')
            globalvars_file.write('        .actv_function =  '+ (self.activation_function).name +',\n        },\n')
    
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

    def generate_flowfacts_dict(self, version):
        
        self.version = version

        if self.version == 'v1':
            
        
            values_loop_filters = ['for loop', 'f', 'nb_filters', 0, (self.nb_filters - 1), []]
            values_loop_output_height = ['for loop', 'i', 'output_height', 0, (self.output_height - 1), []]
            values_loop_output_width = ['for loop', 'j', 'output_width', 0, (self.output_width - 1), []]
            values_loop_input_channels = ['for loop', 'c', 'input_channels', 0, (self.input_channels - 1), []]
            values_loop_kernel_size_1 = ['for loop', 'm', 'kernel_size', 0, (self.kernel_size - 1), []]
            values_loop_kernel_size_2 = ['for loop', 'n', 'kernel_size', 0, (self.kernel_size - 1)]

            self.list_of_values_loops = [values_loop_filters, values_loop_output_height, values_loop_output_width, values_loop_input_channels, values_loop_kernel_size_1, values_loop_kernel_size_2]

            loop_filters = {}
            loop_output_height = {}
            loop_output_width = {}
            loop_input_channels = {}
            loop_kernel_size_1 = {}
            loop_kernel_size_2 = {}
                
            self.list_of_dicts = [loop_filters, loop_output_height, loop_output_width, loop_input_channels, loop_kernel_size_1, loop_kernel_size_2]
            
            return Layers.generate_flowfacts_dict(self)
        
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

    def write_to_layer_c_files(self, data_type, version, layers_source_file, layers_header_file):

        if version == 'v1':

            layers_source_file.write('int ' + self.name + '(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output) \n{ \n')

            layers_source_file.write(self.declare_local_vars(data_type))
            
            layers_source_file.write('    for (int c = 0; c < net[layer_idx].input_channels; ++c)\n    {\n')
            layers_source_file.write('        for (int i = 0; i < net[layer_idx].output_height; ++i)\n        {\n')
            layers_source_file.write('            for (int j = 0; j < net[layer_idx].output_width; ++j)\n            {\n')
            
            layers_source_file.write('                ' + self.update_local_vars())
            
            layers_source_file.write('                for (int m = 0; m < net[layer_idx].pool_size; ++m)\n                {\n')
            layers_source_file.write('                    for (int n = 0; n < net[layer_idx].pool_size; ++n)\n                    {\n')
            layers_source_file.write('                        int ii = i*net[layer_idx].strides + m - net[layer_idx].pad_left;\n')
            layers_source_file.write('                        int jj = j*net[layer_idx].strides + n - net[layer_idx].pad_top;\n\n')
            layers_source_file.write('                        if (ii >= 0 && ii < net[layer_idx].input_height && jj >= 0 && jj < net[layer_idx].input_width)\n                        {\n')

            layers_source_file.write('            ' + self.specific_function(version, '(ii*net[layer_idx].input_width + jj)*net[layer_idx].input_channels + c', 'input'))
            layers_source_file.write('                        }\n                    }\n                }\n')

            layers_source_file.write('            ' + self.generate_output_str('(i*net[layer_idx].output_width + j)*net[layer_idx].input_channels + c', 'output'))
            layers_source_file.write('            }\n        }\n    }\n\n    return 0;\n}\n\n')
            
            layers_header_file.write('int ' + self.name + '(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output);\n')

    def generate_output_str(self, index, output):
        
        return '    '+output+'['+index+'] = '+ self.output_var +';\n\n'

    @abstractmethod    
    def specific_function(self, version, index, input_of_layer):
        pass

    def write_to_function_source_file(self, data_type, version, source_file):
 
        if version == 'v2':
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

            source_file.write(self.specific_function(version, '(ii*'+str(self.input_width)+' + jj)*'+str(self.input_channels)+' + c', 'output_pre'))
            source_file.write('                        }\n                    }\n                }\n')
            source_file.write('            ' + self.generate_output_str('(i*'+str(self.output_width)+' + j)*'+str(self.input_channels)+' + c', 'output_cur'))
            source_file.write('            }\n        }\n    }\n\n')
      
        elif version == 'v3':    
            if self.idx == 1: input_of_layer = 'nn_input'
            else: input_of_layer = 'output_pre'

            source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
            for c in range(self.input_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        source_file.write(self.update_local_vars() )
                        
                        for m in range(self.pool_size):
                            for n in range(self.pool_size):
                                
                                ii = i*self.strides + m - self.pad_left
                                jj = j*self.strides + n - self.pad_top                                
                                
                                if ii >= 0 and ii <self.input_height and jj >= 0 and jj < self.input_width :
                                    
                                    source_file.write(self.specific_function(version, str((ii*self.input_width + jj)*self.input_channels+c), input_of_layer))
 
                                else:
                                    continue

                        source_file.write(self.generate_output_str(str((i*self.output_width+ j)*self.input_channels+ c), 'output_cur') + '\n')

        else:
            pass    

    def write_to_function_header_file(self, version, header_file):
        
        if version == 'v1':

            header_file.write('#define l'+str(self.idx)+'_size ' + str(self.size) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_right ' + str(self.pad_right) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_left ' + str(self.pad_left) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_bottom ' + str(self.pad_bottom) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pad_top ' + str(self.pad_top) + '\n')
            header_file.write('#define l'+str(self.idx)+'_strides ' + str(self.strides) + '\n')
            header_file.write('#define l'+str(self.idx)+'_pool_size ' + str(self.pool_size) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_height ' + str(self.input_height) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_width ' + str(self.input_width) + '\n')
            header_file.write('#define l'+str(self.idx)+'_input_channels ' + str(self.input_channels) + '\n')
            header_file.write('#define l'+str(self.idx)+'_output_height ' + str(self.output_height) + '\n')
            header_file.write('#define l'+str(self.idx)+'_output_width ' + str(self.output_width) + '\n\n')

    def write_to_globalvars_file(self, version, data_type, globalvars_file):

        if version == 'v1':
            
            globalvars_file.write('    ['+str(self.idx)+'] = {\n')
            globalvars_file.write('        .layer_type = '+ self.name +',\n')
            globalvars_file.write('        .layer_size = l'+str(self.idx)+'_size,\n'   )
            globalvars_file.write('        .pad_right = l'+str(self.idx)+'_pad_right,\n')
            globalvars_file.write('        .pad_left = l'+str(self.idx)+'_pad_left,\n')
            globalvars_file.write('        .pad_bottom = l'+str(self.idx)+'_pad_bottom,\n')
            globalvars_file.write('        .pad_top = l'+str(self.idx)+'_pad_top,\n')
            globalvars_file.write('        .strides = l'+str(self.idx)+'_strides,\n')
            globalvars_file.write('        .pool_size = l'+str(self.idx)+'_pool_size,\n')
            globalvars_file.write('        .kernel_size = 0x0,\n')
            globalvars_file.write('        .dilation_rate = 0x0,\n')
            globalvars_file.write('        .nb_filters = 0x0,\n')
            globalvars_file.write('        .input_height = l'+str(self.idx)+'_input_height,\n')
            globalvars_file.write('        .input_width = l'+str(self.idx)+'_input_width,\n')
            globalvars_file.write('        .input_channels = l'+str(self.idx)+'_input_channels,\n')
            globalvars_file.write('        .output_height = l'+str(self.idx)+'_output_height,\n')
            globalvars_file.write('        .output_width = l'+str(self.idx)+'_output_width,\n')
            globalvars_file.write('        .weights = 0x0,\n')
            globalvars_file.write('        .biases = 0x0,\n')
            globalvars_file.write('        .actv_function = 0x0,\n        },\n')

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

    def generate_flowfacts_dict(self, version):
        
        self.version = version

        if self.version == 'v1':
        
            values_loop_input_channels = ['for loop', 'c', 'input_channels', 0, (self.input_channels - 1), []]
            values_loop_output_height = ['for loop', 'i', 'output_height', 0, (self.output_height - 1), []]
            values_loop_output_width = ['for loop', 'j', 'output_width', 0, (self.output_width - 1), []]
            values_loop_pool_size_1 = ['for loop', 'm', 'pool_size', 0, (self.pool_size - 1), []]
            values_loop_pool_size_2 = ['for loop', 'n', 'pool_size', 0, (self.pool_size - 1)]

            self.list_of_values_loops = [values_loop_input_channels, values_loop_output_height, values_loop_output_width, values_loop_pool_size_1, values_loop_pool_size_2]

            loop_output_height = {}
            loop_output_width = {}
            loop_input_channels = {}
            loop_pool_size_1 = {}
            loop_pool_size_2 = {}
                
            self.list_of_dicts = [loop_input_channels, loop_output_height, loop_output_width, loop_pool_size_1, loop_pool_size_2]
            
            return Layers.generate_flowfacts_dict(self)

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

    def specific_function(self, version, index, input_of_layer):
        # Computes the average in this subclass AveragePooling2D 

        if version == 'v3':            
            s = '    '+self.local_var+' += '+input_of_layer+'['+index+'];\n'
            s += '    '+self.local_var_2+' ++;\n'
            
       
        else:
            
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

    def specific_function(self, version, index, input_of_layer):
        
        if version == 'v3':       
            s = '    if ('+input_of_layer+'['+index+'] > '+self.local_var+') '+self.local_var+' = '+input_of_layer+'['+index+'];\n'

        else:

            s = '                            if ('+input_of_layer+'['+index+'] > '+self.local_var+')\n'
            s += '                                '+self.local_var+' = '+input_of_layer+'['+index+'];\n'
    
        return s

class Softmax(Layers):

    def __init__(self, idx, size):
        
        super().__init__()
        self.idx = idx
        self.size = size
        self.name = 'Softmax'

    def write_to_layer_c_files(self, data_type, version, layers_source_file, layers_header_file):
        
        if version == 'v1':

            layers_source_file.write('int Softmax(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output) \n{ \n')
            layers_source_file.write('    '+ data_type + ' sum = 0;\n\n')
            layers_source_file.write('    for (int i = 0; i < net[layer_idx].layer_size; ++i) \n')
            layers_source_file.write('        sum += exp(input[i]);\n\n')
            layers_source_file.write('    for (int j = 0; j < net[layer_idx].layer_size; ++j)\n')
            layers_source_file.write('        output[j] = exp(input[j])/sum;\n\n')
            layers_source_file.write('    return 0; \n} \n\n')
            
            layers_header_file.write('int Softmax(int layer_idx, ' + data_type + ' *input, '+ data_type + ' *output);\n')

    def write_to_function_source_file(self, data_type, version, source_file):
        
        if version == 'v2':    
            source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
            source_file.write('    sum = 0;\n\n')
            source_file.write('    for (int i = 0; i < ' + str(self.size) + '; ++i)\n')
            source_file.write('        sum += exp(output_pre[i]);\n\n')
            source_file.write('    for (int j = 0; j < ' + str(self.size) + '; ++j)\n')
            source_file.write('        output_cur[j] = exp(output_pre[j])/sum;\n\n')


        elif version == 'v3':       
            
            source_file.write('    // ' + self.name + '_' + str(self.idx) + '\n')
            source_file.write('    sum = 0;\n')
            for i in range(self.size):
                source_file.write('    sum += exp(output_pre['+ str(i) +']);\n')
            for j in range(self.size):
                source_file.write('    output_cur['+str(j)+'] = exp(output_pre['+str(j)+'])/sum;\n')
            source_file.write('\n')

        else:
            pass 

    def write_to_function_header_file(self, version, header_file):
        
        if version == 'v1':
            
            header_file.write('#define l'+str(self.idx)+'_size ' + str(self.size) + '\n\n')

        return

    def write_to_globalvars_file(self, version, data_type, globalvars_file):

        if version == 'v1':
  
            globalvars_file.write('    ['+str(self.idx)+'] = {\n')
            globalvars_file.write('        .layer_type = Softmax,\n')
            globalvars_file.write('        .layer_size = l'+str(self.idx)+'_size,\n')
            globalvars_file.write('        .pad_right = 0x0,\n')
            globalvars_file.write('        .pad_left = 0x0,\n')
            globalvars_file.write('        .pad_bottom = 0x0,\n')
            globalvars_file.write('        .pad_top = 0x0,\n')
            globalvars_file.write('        .strides = 0x0,\n')
            globalvars_file.write('        .pool_size = 0x0,\n')
            globalvars_file.write('        .kernel_size = 0x0,\n')
            globalvars_file.write('        .dilation_rate = 0x0,\n')
            globalvars_file.write('        .nb_filters = 0x0,\n')
            globalvars_file.write('        .input_height = 0x0,\n')
            globalvars_file.write('        .input_width = 0x0,\n')
            globalvars_file.write('        .input_channels = 0x0,\n')
            globalvars_file.write('        .output_height = 0x0,\n')
            globalvars_file.write('        .output_width = 0x0,\n')
            globalvars_file.write('        .weights = 0x0,\n')
            globalvars_file.write('        .biases = 0x0,\n')
            globalvars_file.write('        .actv_function = 0x0\n')
            globalvars_file.write('        },\n')

        return

    def feedforward(self, input):
        
        exp = np.exp(input, dtype=np.float)
        output = exp/np.sum(exp)

        return output
    
    def generate_flowfacts_dict(self, version):
        
        self.version = version

        if self.version == 'v1':
            
        
            values_1st_loop = ['for loop', 'i', ('l'+str(self.idx)+'_size'), 0, (self.size - 1)]
            values_2nd_loop = ['for loop', 'j', ('l'+str(self.idx)+'_size'), 0, (self.size - 1)]

            self.list_of_values_loops = [values_1st_loop, values_2nd_loop]

            first_loop = {}
            second_loop = {}

            self.list_of_dicts = [first_loop, second_loop]

            return Layers.generate_flowfacts_dict(self)
