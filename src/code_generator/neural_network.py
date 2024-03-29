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
from code_generator.layers.Softmax import Softmax
from code_generator.layers.MatMul import MatMul
from code_generator.layers.Resize_layers.ResizeLinear import ResizeLinear
from code_generator.layers.Resize_layers.ResizeNearest import ResizeNearest
from code_generator.layers.Resize_layers.ResizeCubic import ResizeCubic
from code_generator.layers.Conv_layers.Conv2D_6loops import Conv2D_6loops
from code_generator.layers.Conv_layers.Conv2D_indirect_gemm import Conv2D_indirect_gemm
from code_generator.layers.Conv_layers.Conv2D_std_gemm import Conv2D_std_gemm
from code_generator.layers.Conv_layers.Conv2D import Conv2D
from code_generator.layers.Concatenate import Concatenate
from code_generator.layers.Pooling_layers.AveragePooling2D import AveragePooling2D
from code_generator.layers.Pooling_layers.MaxPooling2D import MaxPooling2D
from code_generator.layers.Broadcast_layers.Add import Add

class CodeGenerator(ABC):

    def __init__(self, file, test_dataset_file = None, function_name = 'inference', nb_tests = None, conv_algorithm = 'conv_gemm_optim', normalize = False,**kwargs):

        self.file = file
        self.test_dataset_file = test_dataset_file
        self.function_name = function_name
        self.nb_tests = nb_tests
        self.conv_algorithm = conv_algorithm
        self.normalize = normalize

        if (not self.normalize):
            l, dtype, dtype_py, listRoad, maxRoad, dict_cst = parser(self.file, self.conv_algorithm)
        elif(self.normalize):
            l, dtype, dtype_py, listRoad, maxRoad, dict_cst, Normalizer = parser(self.file, self.conv_algorithm, self.normalize)
            self.Normalizer = Normalizer
        
        self.layers = l
        self.data_type = dtype
        self.data_type_py = dtype_py
        self.maxRoad = maxRoad
        self.listRoad = listRoad
        self.dict_cst = dict_cst
        self.data_format = 'channels_first'

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

                if(self.data_format == 'channels_last'): nn_input = np.transpose(np.reshape(nn_input,self.layers[0].input_shape), (2,0,1))

                if (self.normalize):
                    nn_input = self.Normalizer.pre_processing(nn_input)

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

                if(self.data_format == 'channels_last'): nn_output = np.transpose(nn_output, (1,2,0))

                nn_output = np.reshape(nn_output, -1)

                if(self.normalize):
                    nn_output = self.Normalizer.post_processing(nn_output)
                
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
    
    def flatten_array(self,array):
        s = '\n        {'
        shape = array.shape
        for j in range(shape[3]):
            for k in range(shape[0]):
                for f in range(shape[1]):
                    for i in range(shape[2]):
                        s+= str(array[k,f,i,j])+', '
        s = s[:-2]
        s+='}'
        return s

    def generate_testdataset_files(self):

        testdataset_header = open(self.c_files_directory + '/test_dataset.h' , "w+")
        testdataset_source = open(self.c_files_directory + '/test_dataset.c' , "w+")

        with open('src/templates/template_test_dataset_header.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        testdataset_header.write(pystache.render(template, {'nb_tests':self.nb_tests, 'nb_inputs':self.layers[0].size, 'nb_outputs':self.layers[-1].size, 'data_type':self.data_type}))

        dataset = '{'
        if self.test_dataset is None:
            pass
        else:
            for j in range(self.test_dataset.shape[0]):
                dataset += self.flatten_array_orderc(self.test_dataset[j]) +','
            dataset = dataset[:-1]
        
        dataset += '};\n'

        with open('src/templates/template_test_dataset_source.c.tpl','r') as template_file:
            template = template_file.read()
        template_file.close()

        testdataset_source.write(pystache.render(template,{'data_type':self.data_type, 'dataset':dataset}))

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

        mustach_hash = {}

        mustach_hash['data_type'] = self.data_type
        mustach_hash['output_size'] = self.layers[-1].size
        mustach_hash['input_size'] = self.layers[0].size
        mustach_hash['road'] = self.layers[-1].road

        self.l_size_max = 1
        for layer in (self.layers):
            if(hasattr(layer, 'data_format') and layer.data_format=='channels_last'): 
                self.layers[0].data_format = 'channels_last'
                if(isinstance(self.layers[-1], Conv2D) or isinstance(self.layers[-1],MaxPooling2D) or isinstance(self.layers[-1],AveragePooling2D)):
                    mustach_hash['channels_last'] = True
                    mustach_hash['output_channels'] = self.layers[-1].output_channels
                    mustach_hash['output_height'] = self.layers[-1].output_height
                    mustach_hash['output_width'] = self.layers[-1].output_width
            if layer.size > self.l_size_max : self.l_size_max = layer.size
        
        if any((isinstance(layer, Dense) or isinstance(layer,MatMul)) for layer in self.layers):
            mustach_hash['is_dense'] = True
        
        if (any(isinstance(layer, Conv2D_6loops) or isinstance(layer, AveragePooling2D) or isinstance(layer, Softmax)) for layer in self.layers):
            mustach_hash['is_sum'] = True
        
        if any(isinstance(layer, MaxPooling2D) for layer in self.layers):
            mustach_hash['is_max'] = True
               
        if any(isinstance(layer, AveragePooling2D) for layer in self.layers):
            mustach_hash['is_count'] = True
            
        if any(isinstance(layer,ResizeLinear) or isinstance(layer,ResizeCubic) or isinstance(layer,ResizeNearest) for layer in self.layers):
            mustach_hash['is_resize'] = True
        
        if any(isinstance(layer, ResizeCubic) for layer in self.layers):
            mustach_hash['is_cubic_interpolation'] = True
            
        if any(isinstance(layer, ResizeLinear) for layer in self.layers):
            mustach_hash['is_linear_interpolation'] = True
        
        mustach_hash['layers'] = []
        for layer in self.layers:
            layer_hash = {'inference_function':layer.write_to_function_source_file(), 'road':layer.road, 'size':layer.size}
            
            if(layer in self.dict_cst):
                layer_hash['cst'] = True

            mustach_hash['layers'].append(layer_hash)

        with open('src/templates/template_source_file.c.tpl', 'r') as template_file:
            template = template_file.read()
        template_file.close()

        if(self.normalize):
            mustach_hash['pre_processing'] = self.Normalizer.write_pre_processing()
            mustach_hash['post_processing'] = self.Normalizer.write_post_processing()
        
        self.source_file.write(pystache.render(template,mustach_hash))
        
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
            
        if (any(isinstance(layer, Concatenate) or any(isinstance(layer, Conv2D)) or any(isinstance(layer, Dense)) or any(isinstance(layer, Add))) for layer in self.layers):
            mustach_hash['tensor_temp'] = True
            mustach_hash['temp_size'] = max(self.l_size_max,self.patches_size_max)
 
        mustach_hash['layers'] = []
        for layer in self.layers:
            to_print = False
            layer_hash = {'name':layer.name, 'idx':"{:02d}".format(layer.idx)}

            if hasattr(layer, 'weights'):
                layer_hash['nb_weights']=layer.nb_weights
                if layer.nb_weights > self.nb_weights_max : self.nb_weights_max = layer.nb_weights
                to_print = True

            if hasattr(layer,'biases'):
                layer_hash['nb_biases']=layer.nb_biases
                if layer.nb_biases > self.nb_biases_max : self.nb_biases_max = layer.nb_biases
                to_print =True

            if type(layer) is Conv2D_indirect_gemm:
                layer_hash['patches_size'] = layer.patches_size
                to_print = True
            
            if (to_print):
                mustach_hash['layers'].append(layer_hash)
        
        if(self.normalize):
            mustach_hash['normalization_cst'] = self.Normalizer.write_normalization_cst_in_header_file()
        
        with open('src/templates/template_header_file.c.tpl', 'r') as template_file:
            template = template_file.read()
        template_file.close()

        self.header_file.write(pystache.render(template,mustach_hash))
    
    def generate_globalvars_file(self):

        mustach_hash = {}

        mustach_hash['data_type'] = self.data_type
        mustach_hash['road'] = [i for i in range(self.maxRoad)]

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
        
        if (any(isinstance(layer, Concatenate) or any(isinstance(layer, Conv2D)) or any(isinstance(layer, Dense)) or any(isinstance(layer, Add))) for layer in self.layers):
            mustach_hash['tensor_temp'] = True
            mustach_hash['temp_size'] = max(self.l_size_max,self.patches_size_max)
             
        if any(isinstance(layer, Conv2D_indirect_gemm) for layer in self.layers):
            mustach_hash['zero'] = True
        
        mustach_hash['layers'] = []

        for layer in self.layers:
            to_print = False
            layer_hash = {'name':layer.name, 'idx':"{:02d}".format(layer.idx)}

            if hasattr(layer, 'weights'):
                layer_hash['nb_weights'] = layer.nb_weights
                layer_hash['weights'] = self.flatten_array(layer.weights)
                to_print = True

            if hasattr(layer,'biases'):
                layer_hash['nb_biases'] = layer.nb_biases
                layer_hash['biases'] = self.flatten_array_orderc(layer.biases)
                to_print =True

            if type(layer) is Conv2D_indirect_gemm:
                layer_hash['patches_size'] = layer.patches_size
                layer_hash['patches'] = layer.create_ppatches()
                to_print = True
            
            if (to_print):
                mustach_hash['layers'].append(layer_hash)
        
        with open('src/templates/template_global_var_file.c.tpl', 'r') as template_file:
            template = template_file.read()
        template_file.close()

        self.globalvars_file.write(pystache.render(template,mustach_hash))

        if(self.normalize):
            self.globalvars_file.write(self.Normalizer.write_normalization_cst_in_globalvars_file())