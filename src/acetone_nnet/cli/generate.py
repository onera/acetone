"""Entry point for the acetone_generate command line tool.

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
import logging
import pathlib

from acetone_nnet import CodeGenerator


def cli_acetone(
    model_file: str,
    function_name: str,
    nb_tests: int,
    output_dir: str,
    conv_algorithm: str = "std_gemm_nn",
    target: str = "generic",
    target_page_size: int = 4096,
    test_dataset_file: str | None = None,
    *,
    normalize: bool = False,
) -> None:
    """Generate code with ACETONE."""
    logging.info("C CODE GENERATOR FOR NEURAL NETWORKS")
    print(target, " selected")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    net = CodeGenerator(
        file=model_file,
        test_dataset=test_dataset_file,
        function_name=function_name,
        nb_tests=nb_tests,
        normalize=normalize,
        target=target,
        target_page_size=target_page_size,
        versions={"Conv2D": conv_algorithm},
    )
    net.generate_c_files(output_dir)
    net.compute_inference(output_dir)


def acetone_generate() -> None:
    """Generate C code from a neural network model description file."""
    parser = argparse.ArgumentParser(
        description=acetone_generate.__doc__,
    )
    parser.add_argument(
        "--model",
        help="Input file that describes the neural network model",
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--function",
        help="Name of the generated function",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output directory where generated files will be written",
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--dataset-size",
        help="Number of inferences to run (generated or read from test set).",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Input file that contains test data",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--normalize",
        default=False,
        help="Boolean saying if the inputs and outputs needs to be normalized. Only "
        "used when the file is in NNET representation",
        action="store_true",
    )
    parser.add_argument(
        "--conv",
        help="Default variant used for implementation of the Convolution operation.",
        default="std_gemm_nn",
    )

    parser.add_argument(
        "--target",
        help="Target implementation name, default is generic",
        default="generic",
    )

    parser.add_argument(
        "--target_page_size",
        default=4096,
        type=int,
        help="page size of the target in bytes",
    )

    args = parser.parse_args()

    cli_acetone(
        model_file=args.model,
        function_name=args.function,
        output_dir=args.output,
        nb_tests=args.dataset_size,
        conv_algorithm=args.conv,
        target=args.target,
        test_dataset_file=args.dataset,
        normalize=args.normalize,
        target_page_size=args.target_page_size,
    )


if __name__ == "__main__":
    acetone_generate()
