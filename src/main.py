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
import numpy as np
from src.neural_network import CodeGenerator_V1, CodeGenerator_V2, CodeGenerator_V3

def main(model_file, test_dataset_file, function_name, nb_tests, version, output_dir):

    print("CODE GENERATOR FOR NEURAL NETWORKS")

    version_mapping = { 'v1' : CodeGenerator_V1,
                        'v2' : CodeGenerator_V2,
                        'v3' : CodeGenerator_V3}

    codegen_class = version_mapping[version]

    net = codegen_class(json_file = model_file, test_dataset_file = test_dataset_file, function_name = function_name, nb_tests = nb_tests)
    net.generate_c_files(output_dir)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='C code generator for neural networks')

    parser.add_argument("model_file", help="Input file that describes the neural network model")
    parser.add_argument("test_dataset_file", help="Input file that contains test data")
    parser.add_argument("function_name", help="Name of the generated function")
    parser.add_argument("nb_tests", help="Number of inferences process to run")
    parser.add_argument("version", help="Version to be used for the code generation")
    parser.add_argument("output_dir", help="Output directory where generated files will be written")

    args = parser.parse_args()

    main(args.model_file, args.test_dataset_file, args.function_name, args.nb_tests, args.version, args.output_dir)
