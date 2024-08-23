"""Example file to use the debug mode.

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

"""
This file gives a quick tour of the debug mode of acetone-nnet.
The example is using the Onnx debug mode. The process is the same whith Keras.
"""

#### Importing the package ####
import acetone_nnet
import acetone_nnet.debug_tools as debug
import numpy as np

#### Defining the variables ####

target_model = ''  # Replace by the path to the model, or by the model in Onnx (or Keras) (model created/loaded by using their package Python)
dataset = []  # Replace par a numpy array containing the dataset on which to run the debug process (only 1 test)
debug_target = np.array([])  # Replace by a list containing the name of the output to extract. If void, all the intermediate values will be gathered
to_save = False  # Set to True if the new model must be saved
path = ''  # If to_save is True, the model while be save in path (must be a file name)
# Only for onnx debug
otpimize_inputs = False  # Set to True if the initializers treated as inputs are to be fixed (all inputs except the first one will be removed, according to Onnx recommandations)

#### Running the debug mode ####
"""
The three outputs are:
    -'model': The (new) model to parse
    -'targets_indices': The indice the layers at which the outputs are extracted have in acetone
    -'outputs': A list of flattened numpy arry containing the values of the inference
"""
model, targets_indices, outputs_onnx = debug.debug_onnx(target_model=target_model,
                                                        dataset=dataset,
                                                        otpimize_inputs=otpimize_inputs,
                                                        to_save=to_save,
                                                        path=path)

#### Instantiating a 'CodeGenerator' element ####
# Defining the constants to use
model_path = ''  # Replace by the path to the model whose code you want to generate, or directly by the model to parse (either Onnx or Keras model, created using their python package)
test_dataset = None  # Replace by the path to the dataset on which the test must be done, or by a numpy array containing the dataset. If None, the generator while create a random dataset
function_name = ''  # Replace by the name you want to give to the generated function (default is 'inference')
nb_tests = 1  # Don't change this value
conv_algorithm = 'std_gemm_nn'  # Replace by the implementation of the Conv algorithm you desire
normalize = False  # Turn to True if the input model is in format NNet and the normalization option is turned on
output_path_c = ''  # Replace by the path to directory in which the code C while be generated (must be an already existing folder)
output_path_py = ''  # Replace by the path to directory in which the output python while be generated (must be an already existing folder)
debug_mode = 'onnx'  # If the generation must be done in debug mode, replace by the name of the format ('keras' or 'onnx')

# Instantiating a 'CodeGenerator' element
generator = acetone_nnet.CodeGenerator(file=model_path,
                                       test_dataset=test_dataset,
                                       function_name=function_name,
                                       nb_tests=nb_tests,
                                       conv_algorithm=conv_algorithm,
                                       normalize=normalize,
                                       debug_mode=debug_mode)

#### Generating the Python output ####
"""
In debug mode, the output of the function 'compute_inference' change.
It now returns two list:
    -'outputs_python': a list containing flattened numpy array, each being the ouput of a layer. All the intermediary outputs are returned.
    -'targets_python': a list containing the name and the indice of each layer.
"""
outputs_python, targets_python = generator.compute_inference(output_path_py)

"Make sur that the outputs of python are in the same order as the outputs of the model"
outputs_python, targets_python = debug.reorder_outputs_python(outputs_python, targets_python)

"If not all the intermediary outputs are required, use the following line"
# outputs_python, targets_python = debug.extract_outputs_python(generator.compute_inference(output_path_py), targets_indices)


#### Generating the Python output ####
generator.generate_c_files(output_path_c)
"""
Once the file are generated, compile and execute it.
In addition of the usual file, another text file 'debug_file.txt' is produced.
This file contains, one even lines, the name and indices of each layer, as well as several value used to transpose in channels last format. On odd line, it contains the output of the layer
"""
outputs_c, targets_c = debug.extract_outputs_c(output_path_c + "/debug_file.txt",
                                               generator.data_type,
                                               len(generator.debug_target))

#### Comparing the results ####
# Defining the constants to use
verbose = False

# Comparing the result python with the result onnx
same = debug.compare_result(acetone_result=outputs_python,
                            reference_result=outputs_onnx,
                            targets=targets_python,
                            verbose=verbose)

# Comparing the result c with the result onnx
same = debug.compare_result(acetone_result=outputs_c,
                            reference_result=outputs_onnx,
                            targets=targets_python,
                            verbose=verbose)

# Comparing the result python with the result c
same = debug.compare_result(acetone_result=outputs_python,
                            reference_result=outputs_c,
                            targets=targets_python,
                            verbose=verbose)
