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

import argparse
import pathlib

from .code_generator.neural_network import CodeGenerator

def cli_acetone(model_file:str, function_name:str, nb_tests:int, conv_algorithm:str, output_dir:str, test_dataset_file:str|None=None, normalize:bool=False ):
    print("C CODE GENERATOR FOR NEURAL NETWORKS")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    net = CodeGenerator(file = model_file, test_dataset_file = test_dataset_file, function_name = function_name, nb_tests = nb_tests, conv_algorithm = conv_algorithm, normalize = normalize)
    net.generate_c_files(output_dir)
    net.compute_inference(output_dir)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='C code generator for neural networks')

    parser.add_argument("model_file", help="Input file that describes the neural network model")
    parser.add_argument("function_name", help="Name of the generated function")
    parser.add_argument("nb_tests", help="Number of inferences process to run")
    parser.add_argument("conv_algorithm", help="Algorithm to be used in convolutional layer. Default is indirect im2col with GeMM")
    parser.add_argument("output_dir", help="Output directory where generated files will be written")
    parser.add_argument("test_dataset_file", nargs='?', default=None, help="Input file that contains test data")
    parser.add_argument("normalize", nargs='?', default=False, help="Boolean saying if the inputs and outputs needs to be normalized. Only used when the file is in NNET representation")

    args = parser.parse_args()

    cli_acetone(args.model_file, args.function_name, args.nb_tests, args.conv_algorithm, args.output_dir, args.test_dataset_file, args.normalize)
