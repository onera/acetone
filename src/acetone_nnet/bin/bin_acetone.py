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

from ..cli_acetone import cli_acetone
from ..cli_compare import cli_compare

def acetone_generate():
    parser = argparse.ArgumentParser(description='C code generator for neural networks')

    parser.add_argument("model_file", help="Input file that describes the neural network model")
    parser.add_argument("function_name", help="Name of the generated function")
    parser.add_argument("nb_tests", help="Number of inferences process to run")
    parser.add_argument("conv_algorithm", help="Algorithm to be used in convolutional layer. Default is indirect im2col with GeMM")
    parser.add_argument("output_dir", help="Output directory where generated files will be written")
    parser.add_argument("test_dataset_file", nargs='?', default=None, help="Input file that contains test data")
    parser.add_argument("normalize", nargs='?', default=False, help="Boolean saying if the inputs and outputs needs to be normalized. Only used when the file is in NNET representation")

    args = parser.parse_args()

    cli_acetone(model_file=args.model_file, 
                function_name=args.function_name, 
                nb_tests=args.nb_tests, 
                conv_algorithm=args.conv_algorithm, 
                output_dir=args.output_dir,
                test_dataset_file=args.test_dataset_file,
                normalize=args.normalize)

def acetone_compare():
    parser = argparse.ArgumentParser(description='Program to verify the semantic preservation of ')

    parser.add_argument("reference_file", help="File with the inference output of the reference machine learning framework")
    parser.add_argument("c_file", help="File with the inference output of the studied machine learning framework")
    parser.add_argument("nb_tests", help="Number of inferences process to compare")
    parser.add_argument("--precision", help="Precision of the data studied. Default is float32")

    args = parser.parse_args()

    if args.precision:
        precision = args.precision
    else:
        precision = 'float'

    cli_compare(reference_file=args.reference_file,
                c_file=args.c_file,
                nb_tests=args.nb_tests,
                precision=precision)