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

import os
import json
import numpy as np
from pathlib import Path
from itertools import islice
from src.code_generator.activation_functions import Linear, ReLu, Sigmoid, TanH
from src.code_generator.layers import AveragePooling2D, MaxPooling2D, InputLayer, Dense, Conv2D, Softmax
from abc import ABC

class CodeGenerator(ABC):

    def __init__(self, json_file, test_dataset_file = None, function_name = 'inference', nb_tests = None, **kwds):

        self.json_file = json_file
        self.test_dataset_file = test_dataset_file
        self.function_name = function_name
        self.nb_tests = nb_tests

        l, dtype, dtype_py = self.load_json()
        self.layers = l
        self.data_type = dtype
        self.data_type_py = dtype_py

        ds = self.load_test_dataset()
        self.test_dataset = ds

        self.files_to_gen = ['inference.c', 'inference.h', 'global_vars.c', 'main.c', 'Makefile']
        
    def load_json(self):
        
        file = open(self.json_file, 'r')
        model = json.load(file)

        data_type = model['config']['layers'][0]['config']['dtype']

        if data_type == 'float64':
            data_type = 'double'
            data_type_py = np.float64

        elif data_type == 'float32':
            data_type = 'float'
            data_type_py = np.float32

        elif data_type == 'int':
            data_type = 'long int'
            data_type_py = np.int32

        layers = []

        l_temp = InputLayer(0, model['config']['layers'][0]['config']['size'])

        layers.append(l_temp)

        nb_softmax_layers = 0
        nb_flatten_layers = 0

        for idx, layer in list(islice(enumerate(model['config']['layers']), 1, None)):

            idx += nb_softmax_layers
            idx -= nb_flatten_layers

            if 'activation' in layer['config']:
                if layer['config']['activation'] == 'softmax':
                    layer['config']['activation'] = 'linear'
                    add_softmax_layer = True
                else:
                    add_softmax_layer = False
            else:
                pass
        
            if layer['class_name'] == 'Dense':
                current_layer = Dense(idx, layer['config']['units'], data_type_py(layer['weights']), data_type_py(layer['biases']), self.create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'Conv2D': 
                current_layer = Conv2D(idx, layer['config']['size'], layer['config']['padding'], layer['config']['strides'][0], layer['config']['kernel_size'][0], layer['config']['dilation_rate'][0], layer['config']['filters'], layer['config']['input_shape'], layer['config']['output_shape'], data_type_py(layer['weights']), data_type_py(layer['biases']), self.create_actv_function_obj(layer['config']['activation']))
            
            elif layer['class_name'] == 'AveragePooling2D':
                current_layer = AveragePooling2D(idx = idx, size = layer['config']['size'], padding = layer['config']['padding'], strides = layer['config']['strides'][0], pool_size = layer['config']['pool_size'][0], input_shape = layer['config']['input_shape'], output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'MaxPooling2D':
                current_layer = MaxPooling2D(idx = idx, size = layer['config']['size'], padding = layer['config']['padding'], strides = layer['config']['strides'][0], pool_size = layer['config']['pool_size'][0], input_shape = layer['config']['input_shape'], output_shape = layer['config']['output_shape'])
            
            elif layer['class_name'] == 'Flatten':
                nb_flatten_layers = 1
                continue
            
            l_temp.next_layer.append(current_layer)
            current_layer.previous_layer.append(l_temp)
            l_temp = current_layer
            layers.append(current_layer)

            # Separeted method to generate softmax
            if add_softmax_layer:
                nb_softmax_layers += 1
                current_layer = Softmax(idx+1, l_temp.size)
                l_temp.next_layer.append(current_layer)
                current_layer.previous_layer.append(l_temp)
                l_temp = current_layer
                layers.append(current_layer)

        print("Finished model initialization.")    
        return layers, data_type, data_type_py

    def load_test_dataset(self):
        
        test_dataset = []
        try:
            with open(self.test_dataset_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line[1:-2].split(',')
                    if self.data_type == 'int':
                        line = list(map(int,line))
                    elif self.data_type == 'double':
                        line = list(map(float,line)) 
                    elif self.data_type == 'float':
                        line = list(map(np.float32,line))
                    test_dataset.append(line)
                    if i == int(self.nb_tests)-1:
                        break
            test_dataset = np.array(test_dataset)
            f.close()
       
        except TypeError:
            None
        
        return test_dataset

    def compute_inference(self, c_files_directory):
        with open(os.path.join(c_files_directory, 'output_python.txt'), 'w+') as fi:
            for nn_input in self.test_dataset:
            
                previous_layer_result = nn_input  # for the very first layer, it is the neural network input
                
                for layer in self.layers:
                    current_layer_result = layer.feedforward(previous_layer_result)
                    previous_layer_result = current_layer_result
                
                nn_output = current_layer_result
        
                # print(nn_output) # write to file instead
                
                # Write results in text files to compare prediction.
                
                nn_output = np.reshape(nn_output, -1)
                for j in range(len(nn_output)):
                    print('{:.9g}'.format(nn_output[j]), end=' ', file=fi, flush=True)           
                    # print(decimal.Decimal(nn_output[j]), end=' ', file=fi, flush=True)
                print(" ",file=fi)
        
        fi.close()

        print("File output_python.txt generated.")

        return nn_output

    def create_actv_function_obj(self, activation_str):

        if activation_str == 'sigmoid':
            return Sigmoid()
        elif activation_str == 'relu':
            return ReLu()
        elif activation_str == 'tanh':
            return TanH()
        elif activation_str == 'linear':
            return Linear()
        elif activation_str == 'softmax':
            return Softmax()

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
        
        if ndim > 2:
            array = array.reshape(-1, *array.shape[-(ndim-2):])
            flattened_aray = array.flatten(order='F')

        else:
            flattened_aray = array.flatten(order='F')

        s = '\n        {'
        for i in range(flattened_aray.size):
            s += str(flattened_aray[i])+', '
        s = s[:-2]
        s+='}'
        
        return s

    def generate_testdataset_files(self):

        testdataset_header = open(self.c_files_directory + '/test_dataset.h' , "w+")
        testdataset_source = open(self.c_files_directory + '/test_dataset.c' , "w+")

        s = '#ifndef TEST_DATASET_H_ \n'
        s += '#define TEST_DATASET_H_ \n\n'
        s += '#define nb_samples ' + str(self.nb_tests) + '\n'
        s += '#define nn_input_size ' + str(self.layers[0].size) + '\n'
        s += '#define nn_output_size ' + str(self.layers[-1].size) + '\n\n' # last element of layers_sizes, corresponding to the size of last layer.
        s += 'extern '+ self.data_type + ' nn_test_inputs[nb_samples][nn_input_size];\n\n'
        s += '#endif'

        testdataset_header.write(s)

        t = '#include "test_dataset.h" \n\n'
        t += self.data_type + ' nn_test_inputs[nb_samples][nn_input_size] = {'

        if self.test_dataset is None:
            pass
        else:
            for j in range(self.test_dataset.shape[0]):
                t += self.flatten_array_orderc(self.test_dataset[j]) +','
            t = t[:-1]
        
        t += '};\n'

        testdataset_source.write(t)

    def generate_main_file(self):

        self.main_file.write('#include <stdio.h> \n#include <math.h> \n#include <time.h> \n#include "test_dataset.h" \n#include "inference.h"\n\n')
        self.main_file.write('struct timeval GetTimeStamp();\n\n')
        self.main_file.write('int main(int argc, char** argv)\n{\n')
        self.main_file.write('    char *path = argv[1];\n\n')
        self.main_file.write('    FILE *fp = fopen(path, "w+");\n\n')
        self.main_file.write('    '+self.data_type+' predictions[nb_samples][nn_output_size];\n\n')
        self.main_file.write('    clock_t t0 = clock();\n')
        self.main_file.write('    for (int i = 0; i < nb_samples; ++i){\n')
        self.main_file.write('        inference(predictions[i], nn_test_inputs[i]);\n    }\n')
        self.main_file.write('    clock_t t1 = clock();\n\n')
        self.main_file.write('    printf("   Average time over %d tests: %e s \\n", nb_samples,\n')
        self.main_file.write('        (float)(t1-t0)/(float)CLOCKS_PER_SEC/(float)100);\n\n')
        self.main_file.write('    printf("   ACETONE framework\'s inference output: \\n");\n')
        self.main_file.write('    for (int i = 0; i < nb_samples; ++i){\n')
        self.main_file.write('        for (int j = 0; j < nn_output_size; ++j){\n')
        self.main_file.write('            fprintf(fp,"%.9g ", predictions[i][j]);\n')
        self.main_file.write('            printf("%.9g ", predictions[i][j]);\n')
        self.main_file.write('            if (j == nn_output_size - 1){\n')
        self.main_file.write('                fprintf(fp, "\\n");\n')
        self.main_file.write('                printf("\\n");\n')
        self.main_file.write('            }\n        }\n    }\n\n')
        self.main_file.write('    fclose(fp);\n')
        self.main_file.write('    fp = NULL;\n\n')
        self.main_file.write('    return 0;\n}')

    def generate_makefile(self):

        header_files = []
        source_files = []
        for filename in self.files_to_gen:
            if '.c' in filename : source_files.append(filename)
            elif '.h' in filename : header_files.append(filename)
            else : pass

        self.makefile.write('CC = gcc\n')
        self.makefile.write('CFLAGS = -g -w -lm\n\n')
        self.makefile.write('SRC = ' + ' '.join(source_files) + '\n')
        self.makefile.write('HEADERS = ' + ' '.join(header_files) + '\n')
        self.makefile.write('OBJ = $(SRC:.cc=.o) $(HEADERS)\n')
        self.makefile.write('EXEC = '+ self.function_name +'\n\n')
        self.makefile.write('all: $(EXEC)\n\n')
        self.makefile.write('$(EXEC): $(OBJ)\n')
        self.makefile.write('	$(CC) $(LDFLAGS)  -o $@ $(OBJ) $(LBLIBS) $(CFLAGS)\n\n')
        self.makefile.write('clean:\n	rm $(EXEC)')

    def generate_c_files(self, c_files_directory):  

        self.c_files_directory = c_files_directory

        testdataset_files = ['test_dataset.h', 'test_dataset.c']
        self.files_to_gen.extend(testdataset_files)
        
        q = 0
        for file in self.files_to_gen:
            filename = Path(os.path.join(self.c_files_directory, file))

            if filename.is_file():
                print("ERROR : " + file + " already exists !")
                q += 1
            
        if q != 0 : exit()
        
        else:                               
            self.source_file = open(self.c_files_directory + '/inference.c' , "a+")
            self.header_file = open(self.c_files_directory + '/inference.h' , "a+")
            self.globalvars_file = open(self.c_files_directory + '/global_vars.c' , "a+")
            self.main_file = open(self.c_files_directory + '/main.c' , "a+")
            self.makefile = open(self.c_files_directory + '/Makefile' , "a+")

        self.generate_function_source_file()
        print('Generated function source file.') 
        self.generate_function_header_file()
        print('Generated function header file.') 
        self.generate_globalvars_file()
        print('Generated globalvars .c file.') 
        self.generate_main_file()
        print('Generated main file.')
        self.generate_makefile()
        print('Generated Makefile.')
        self.generate_testdataset_files()
        print('Generated testdataset files.')
      
    def generate_function_source_file(self):

        self.l_size_max = 1
        for layer in (self.layers):
            if layer.size > self.l_size_max : self.l_size_max = layer.size            

        self.source_file.write('#include <stdio.h>\n')
        self.source_file.write('#include <math.h>\n')
        self.source_file.write('#include "inference.h"\n\n')

        self.source_file.write('int inference(' + self.data_type + ' prediction[' + str(self.layers[-1].size) + '], ' + self.data_type + ' nn_input[' + str(self.layers[0].size) + '])\n{\n')
        self.source_file.write('    static ' + self.data_type + ' output_pre[' + str(self.l_size_max) + '];\n')
        self.source_file.write('    static ' + self.data_type + ' output_cur[' + str(self.l_size_max) + '];\n')
        self.source_file.write('    ' + self.data_type + ' dotproduct;\n')
        self.source_file.write('    ' + self.data_type + ' sum;\n')
        self.source_file.write('    ' + self.data_type + ' max;\n')
        self.source_file.write('    int count;\n\n')

        for layer in self.layers:

            layer.write_to_function_source_file(self.data_type, self.source_file)
            
            if layer.idx > 0:
                self.source_file.write('    for (int k = 0; k < ' + str(layer.size) + '; ++k)\n')

                if layer.idx == len(self.layers)-1:
                    self.source_file.write('        prediction[k] = output_cur[k];\n\n')
                else:
                    self.source_file.write('        output_pre[k] = output_cur[k];\n\n')
            
        self.source_file.write('    return 0;\n}')

    def generate_function_header_file(self):

        self.header_file.write('#ifndef INFERENCE_H_ \n')
        self.header_file.write('#define INFERENCE_H_ \n\n')

        self.nb_weights_max = 1
        self.nb_biases_max = 1

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.header_file.write('extern '+ self.data_type + ' weights_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_weights) + '];\n')
                self.header_file.write('extern '+ self.data_type + ' biases_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_biases) + '];\n')
                if layer.nb_weights > self.nb_weights_max : self.nb_weights_max = layer.nb_weights
                if layer.nb_biases > self.nb_biases_max : self.nb_biases_max = layer.nb_biases

        self.header_file.write('\nint inference('+ self.data_type +' *prediction, '+ self.data_type +' *nn_input);\n\n')
        self.header_file.write('#endif')
    
    def generate_globalvars_file(self):

        self.globalvars_file.write('#include "inference.h" \n\n')

        for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.globalvars_file.write(self.data_type + ' weights_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_weights) + '] = ' \
                                            + self.flatten_array_orderc(layer.weights) + ';\n')
                    self.globalvars_file.write(self.data_type + ' biases_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_biases) + '] = ' \
                                            + self.flatten_array_orderc(layer.biases) + ';\n\n')
   