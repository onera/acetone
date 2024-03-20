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
import numpy as np
from pathlib import Path
from abc import ABC
import pystache

from format_importer.parser import parser

from code_generator.layers.Dense import Dense
from code_generator.layers.Resize_layers.ResizeLinear import ResizeLinear
from code_generator.layers.Resize_layers.ResizeNearest import ResizeNearest
from code_generator.layers.Resize_layers.ResizeCubic import ResizeCubic
from code_generator.layers.Conv_layers.Conv2D_6loops import Conv2D_6loops
from code_generator.layers.Conv_layers.Conv2D_indirect_gemm import Conv2D_indirect_gemm
from code_generator.layers.Conv_layers.Conv2D_std_gemm import Conv2D_std_gemm
from code_generator.layers.Concatenate import Concatenate
from code_generator.layers.Pooling_layers.AveragePooling2D import AveragePooling2D
from code_generator.layers.Pooling_layers.MaxPooling2D import MaxPooling2D
from code_generator.layers.Broadcast_layers.Add import Add

class CodeGenerator(ABC):

    def __init__(self, file, test_dataset_file = None, function_name = 'inference', nb_tests = None, conv_algorithm = 'conv_gemm_optim', **kwargs):

        self.file = file
        self.test_dataset_file = test_dataset_file
        self.function_name = function_name
        self.nb_tests = nb_tests
        self.conv_algorithm = conv_algorithm

        l, dtype, dtype_py, listRoad, maxRoad, dict_cst = parser(self.file, self.conv_algorithm)

        self.layers = l
        self.data_type = dtype
        self.data_type_py = dtype_py
        self.maxRoad = maxRoad
        self.listRoad = listRoad
        self.dict_cst = dict_cst

        if test_dataset_file:
            ds = self.load_test_dataset()
            self.test_dataset = ds
        else:
            print("creating random dataset")
            ds = self.create_test_dataset()
            self.test_dataset = ds
            

        self.files_to_gen = ['inference.c', 'inference.h', 'global_vars.c', 'main.c', 'Makefile', 'test_dataset.h', 'test_dataset.c']
        
    def create_test_dataset(self):
        test_dataset = np.float32(np.random.default_rng(seed=10).random((int(self.nb_tests),1,int(self.layers[0].size))))
        return test_dataset 
     
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
                
                previous_layer_result = [nn_input for i in range(self.maxRoad)]  # for the very first layer, it is the neural network input
                
                to_store = {} #a dictionnary containing the values to store
                for layer in self.layers:
                    if(not layer.previous_layer):
                        previous_layer_result[layer.road] = layer.feedforward(previous_layer_result[layer.road]) #if the layer is an input layer, it directly take the vaue from it's road
                    elif(len(layer.previous_layer)==1):
                        if(len(layer.previous_layer[0].next_layer) == 1):
                            previous_layer_result[layer.road] = layer.feedforward(previous_layer_result[layer.previous_layer[0].road]) #if the layer has exactly one previous_layer, it takes the value from it's father's road
                        else:
                            previous_layer_result[layer.road] = layer.feedforward(to_store[layer.previous_layer[0].idx]) #if the father is stored, we take it from the storage
                            layer.previous_layer[0].sorted +=1 #the number of children already "taken care of"
                    else:#if the layer has multiple ancestors, we take all of their value
                        prev_layer = []
                        for prev in layer.previous_layer:
                            if(len(prev.next_layer) == 1):
                                prev_layer.append(previous_layer_result[prev.road])
                            else:
                                prev_layer.append(to_store[prev.idx])
                                prev.sorted +=1
                        previous_layer_result[layer.road] = layer.feedforward(prev_layer)
                    
                    #After having computed the value of the layer, we check if there is a fused layer.
                    if(layer.fused_layer):
                        #If the current layer is the last layer to be updated, the fused layer must be computed
                        if((layer.fused_layer.count_updated_prior_layers == len(layer.fused_layer.prior_layers)-1)):
                            fused = layer.fused_layer
                            #The layer has multiple ancestors (otherwise, it is treated as an activation function)
                            prev_layer = []
                            for prev in fused.prior_layers:
                                if(prev.idx not in to_store):# Taking the value where it is stored
                                    prev_layer.append(previous_layer_result[prev.road])
                                else:
                                    prev_layer.append(to_store[prev.idx])
                                    prev.sorted +=1
                            previous_layer_result[layer.road] = fused.feedforward(prev_layer)
                        #Else, we notify the fused layer that another one of its ancestors have been computed
                        else:
                            layer.fused_layer.count_updated_prior_layers+=1
                    
                    if (len(layer.next_layer) > 1): #if the layer has more than one child, it needs to be stored
                        to_store[layer.idx] = previous_layer_result[layer.road]
                        
                    for prev in layer.previous_layer:
                        if ((prev.sorted == len(prev.next_layer)) and (prev in to_store)):#if all the children of the parent layer are "taken care of", we "forget" the parent's value ( *2 because of the creation of the dict in graph.to_save)
                            to_store.pop(prev.idx)
                            
                nn_output = previous_layer_result[layer.road]
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

        with open('src/templates/template_main_file.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        self.main_file.write(pystache.render(template, {'data_type':self.data_type}))

    def generate_makefile(self):

        header_files = []
        source_files = []
        for filename in self.files_to_gen:
            if '.c' in filename : source_files.append(filename)
            elif '.h' in filename : header_files.append(filename)
            else : pass

        with open('src/templates/template_Makefile.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        self.makefile.write(pystache.render(template, {'source_files':' '.join(source_files), 'header_files':' '.join(header_files), 'function_name':self.function_name}))

    def generate_c_files(self, c_files_directory):  

        self.c_files_directory = c_files_directory
        
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
        # self.source_file.write('    static ' + self.data_type + ' output_cur[' + str(self.l_size_max) + '];\n')
        # self.source_file.write('    static ' + self.data_type + ' output_pre[' + str(self.l_size_max) + '];\n')

        def write_cst_cubic_interpolation():
            #the two points for a 1D interpolation and their values if the non void dimension is the width of the tensor
            s = '    float a;\n'
            s+= '    float result_interpolation;\n'
            s+= '    float f_1;\n'
            s+= '    float f0;\n'
            s+= '    float f1;\n'
            s+= '    float f2;\n'
            s+= '    float s;\n'
            return s
        
        
        def write_cst_linear_interpolation():
            #the four points for a 2D interpolation and their values
            s = '    int y2;\n'
            s+= '    int y1;\n'
            s+= '    int x2;\n'
            s+= '    int x1;\n'
            s+= '    float f11;\n'
            s+= '    float f12;\n'
            s+= '    float f21;\n'
            s+= '    float f22;\n'
            return s
            
        if any(isinstance(layer, Dense) for layer in self.layers):
            self.source_file.write('    ' + self.data_type + ' dotproduct;\n')
        
        if (any(isinstance(layer, Conv2D_6loops) or isinstance(layer, AveragePooling2D)) for layer in self.layers):
            self.source_file.write('    ' + self.data_type + ' sum;\n')
        
        if any(isinstance(layer, MaxPooling2D) for layer in self.layers):
            self.source_file.write('    ' + self.data_type + ' max;\n')
               
        if any(isinstance(layer, AveragePooling2D) for layer in self.layers):
            self.source_file.write('    int count;\n\n')
            
        if any(isinstance(layer,ResizeLinear) or isinstance(layer,ResizeCubic) or isinstance(layer,ResizeNearest) for layer in self.layers):
            self.source_file.write('    float x;\n    float y;\n')
        
        if any(isinstance(layer, ResizeCubic) for layer in self.layers):
            self.source_file.write(write_cst_cubic_interpolation())
            
        if any(isinstance(layer, ResizeLinear) for layer in self.layers):
            self.source_file.write(write_cst_linear_interpolation())
        

        for layer in self.layers:
            layer.write_to_function_source_file(self.source_file)
            
            if(layer in self.dict_cst):
                self.source_file.write('    for (int k = 0; k < ' + str(layer.size) + '; ++k)\n')
                self.source_file.write('    {\n        cst_'+ str(self.dict_cst[layer]) +'[k] = output_'+str(layer.road)+'[k];\n    }\n\n')
            
            if layer == self.layers[-1]:
                self.source_file.write('    for (int k = 0; k < ' + str(layer.size) + '; ++k)\n')
                self.source_file.write('        prediction[k] = output_'+str(layer.road)+'[k];\n\n')
                
        self.source_file.write('    return 0;\n}')

    def write_ouput_in_file(self,str_size,source_file):
        for i in range(self.maxRoad):
            source_file.write('// output list for road ' + str(i)+'\n')
            source_file.write(self.data_type + ' output_'+str(i)+'[' + str_size + '];\n')
        source_file.write('\n')
        
    def generate_function_header_file(self):

        mustach_hash = {}
        mustach_hash['data_type'] = self.data_type
        mustach_hash['road'] = [i for i in range(self.maxRoad)]

        self.nb_weights_max = 1
        self.nb_biases_max = 1

        self.patches_size_max = 1
        self.concate_size_max = 0
        for layer in self.layers: 
            if isinstance(layer, Conv2D_std_gemm):
                if layer.patches_size > self.patches_size_max : self.patches_size_max = layer.patches_size
            if isinstance(layer,Concatenate):
                self.patches_size_max = max(self.patches_size_max,layer.size)
                
        if any(isinstance(layer, Conv2D_std_gemm) for layer in self.layers):
            mustach_hash['road_size'] = max(self.l_size_max,self.patches_size_max)          
        else:
            mustach_hash['road_size'] = self.l_size_max
            
        mustach_hash['cst'] = []
        written = {}
        for layer in self.dict_cst:
            if self.dict_cst[layer] not in written:
                written[self.dict_cst[layer]] = layer.size
            else:
                written[self.dict_cst[layer]] = max(written[self.dict_cst[layer]],layer.size)

        for cst in written:
            mustach_hash['cst'].append({'name':cst, 'size':written[cst]})
            
        if (any(isinstance(layer, Concatenate) or any(isinstance(layer, Conv2D_std_gemm)) or any(isinstance(layer, Dense)) or any(isinstance(layer, Add))) for layer in self.layers):
            mustach_hash['tensor_temp'] = True
            mustach_hash['temp_size'] = max(self.l_size_max,self.patches_size_max)
            
        mustach_hash['layers'] = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                mustach_hash['layers'].append({'name':layer.name, 'idx':"{:02d}".format(layer.idx), 'nb_weights':layer.nb_weights, 'nb_biases':layer.nb_biases})

                if type(layer) is Conv2D_indirect_gemm:
                    mustach_hash['layers'][-1]['patches_size'] = layer.patches_size

                if layer.nb_weights > self.nb_weights_max : self.nb_weights_max = layer.nb_weights
                if layer.nb_biases > self.nb_biases_max : self.nb_biases_max = layer.nb_biases
        
        with open('src/templates/template_header_file.c.tpl', 'r') as template_file:
            template = template_file.read()
        template_file.close()

        self.header_file.write(pystache.render(template,mustach_hash))
    
    def generate_globalvars_file(self):

        self.globalvars_file.write('#include "inference.h" \n\n')

        if any(isinstance(layer, Conv2D_std_gemm) for layer in self.layers):       
            self.write_ouput_in_file(str(max(self.l_size_max,self.patches_size_max)),self.globalvars_file)
        
        else:
            self.write_ouput_in_file(str(self.l_size_max),self.globalvars_file)
        
        
        written = {}
        for layer in self.dict_cst:
            if self.dict_cst[layer] not in written:
                written[self.dict_cst[layer]] = layer.size
            else:
                written[self.dict_cst[layer]] = max(written[self.dict_cst[layer]],layer.size)
        
        for cst in written:
            self.globalvars_file.write(self.data_type + ' cst_'+str(cst)+'[' + str(written[cst]) + '];\n')
        self.globalvars_file.write('\n')
        
        if (any(isinstance(layer, Concatenate) 
                or any(isinstance(layer, Conv2D_indirect_gemm)) 
                or any(isinstance(layer, Conv2D_std_gemm)) 
                or any(isinstance(layer, Dense))  
                or any(isinstance(layer, Add))) 
                for layer in self.layers):
            self.globalvars_file.write(self.data_type + ' tensor_temp[' + str(max(self.l_size_max,self.patches_size_max)) + '];\n\n')
             
        if any(isinstance(layer, Conv2D_indirect_gemm) for layer in self.layers):
            self.globalvars_file.write(self.data_type + ' zero = 0.0f;\n\n')

        

        for layer in self.layers:
                if hasattr(layer, 'weights'):
                    if(("json" in self.file[-4:]) or ("h5" in self.file[-4:]) or ("nnet" in self.file[-4:])):
                        self.globalvars_file.write(self.data_type + ' weights_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_weights) + '] = ' \
                                            + self.flatten_array_hybrid(layer.weights) + ';\n')
                    
                    elif("onnx" in self.file[-4:]):
                        self.globalvars_file.write(self.data_type + ' weights_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_weights) + '] = ' \
                                            + self.flatten_array_orderc(layer.weights) + ';\n')
                    
                    self.globalvars_file.write(self.data_type + ' biases_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.nb_biases) + '] = ' \
                                            + self.flatten_array_orderc(layer.biases) + ';\n\n')
                    
                    if type(layer) is Conv2D_indirect_gemm:
                        self.globalvars_file.write(self.data_type + ' *ppatches_' + layer.name + '_' + str("{:02d}".format(layer.idx)) + '[' + str(layer.patches_height*layer.patches_width) + '] = ' \
                                            + layer.create_ppatches(self.dict_cst) + ';\n\n')
   