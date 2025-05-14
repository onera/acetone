"""Example file to use the package.

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
from acetone_nnet.cli.generate import cli_acetone

"""
This file present an example of how to use the package acetone-nnet
"""

#### Importing the package ####
import acetone_nnet

#### Generating the code ####
"""
There is two way to generate the code:
    -Using the function 'cli_acetone' to directly generate both the output python and the code C
    -Using the class 'CodeGenerator' to have more controle on the generation
"""

## Using 'cli_acetone' ##
"""
This method is mainly used as a command-line, either by runing the python file, either by using the built in command: acetone_generate.
Confere to the ReadMe for example using a terminal.
"""

# Defining the constants to use
model_path = ""  # Replace by the path to the model whose code you want to generate
output_path = ""  # Replace by the path to directory in which the output while be generated (must be an already existing folder)
function_name = ""  # Replace by the name you want to give to the generated function (default is 'inference')
nb_tests = 1  # Replace by the number of tests you want to run
conv_algorithm = "std_gemm_nn"  # Replace by the implementation of the Conv algorithm you desire
test_dataset = None  # Replace by the path to the dataset on which the test must be done. If None, the generator while create a random dataset
normalize = False  # Turn to True if the input model is in format NNet and the normalization option is turned on

# The function while generate the C code and the output of the Python inference is the directory output_path
cli_acetone(model_file=model_path,
            function_name=function_name,
            nb_tests=nb_tests,
            conv_algorithm=conv_algorithm,
            output_dir=output_path,
            test_dataset_file=test_dataset,
            normalize=normalize)

## Using 'CodeGenerator' ##
"""
This method is prefered when using the package. 
It allows more regarding the type of the arguments, give more controle over the generation.
"""
# Defining the constants to use
model_path = ""  # Replace by the path to the model whose code you want to generate, or directly by the model to parse (either Onnx or Keras model, created using their python package)
test_dataset = None  # Replace by the path to the dataset on which the test must be done, or by a numpy array containing the dataset. If None, the generator while create a random dataset
function_name = ""  # Replace by the name you want to give to the generated function (default is 'inference')
nb_tests = 1  # Replace by the number of tests you want to run
conv_algorithm = "std_gemm_nn"  # Replace by the implementation of the Conv algorithm you desire
normalize = False  # Turn to True if the input model is in format NNet and the normalization option is turned on
output_path_c = ""  # Replace by the path to directory in which the code C while be generated (must be an already existing folder)
output_path_py = ""  # Replace by the path to directory in which the output python while be generated (must be an already existing folder)
debug_mode = None  # If the generation must be done in debug mode, replace by the name of the format ('keras' or 'onnx')

# Instantiating a 'CodeGenerator' element
generator = acetone_nnet.CodeGenerator(file=model_path,
                                       test_dataset=test_dataset,
                                       function_name=function_name,
                                       nb_tests=nb_tests,
                                       conv_algorithm=conv_algorithm,
                                       normalize=normalize,
                                       debug_mode=debug_mode)

# Generating the C code
generator.generate_c_files(output_path_c)

# Generating the Python code
generator.compute_inference(output_path_py)
