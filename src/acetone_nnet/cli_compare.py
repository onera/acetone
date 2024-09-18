"""Compare two text file.

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
import sys
from pathlib import Path

import numpy as np

# Source document for floating comparison
# https://floating-point-gui.de/errors/comparison/


def compare_floats(a: float, b: float, epsilon: float = (128 * sys.float_info.epsilon),
                   abs_th: float = sys.float_info.min) -> (bool, float):
    """Compare a-b to abs_th."""
    diff = abs(a - b)
    if a == b:
        return True, diff

    norm = min((abs(a) + abs(b)), sys.float_info.max)
    if diff < max(abs_th, epsilon * norm):
        return True, diff

    return False, diff


def preprocess_line(line: list, precision: str) -> list:
    """Cast a list of str to a list of float."""
    line = line.split(" ")
    line[:] = [x for x in line if x.strip()]
    if precision == "double":
        line = list(map(np.float64, line))
    elif precision == "float":
        line = list(map(np.float32, line))
    return line


def compare_lines(line_f1: list, line_f2: list, precision: str) -> (bool, float):
    """Compare two lines element wise using compare_floats."""
    line_f1 = preprocess_line(line_f1, precision)
    line_f2 = preprocess_line(line_f2, precision)
    max_diff = 0
    length = len(line_f1)
    line_comparison = True

    for j in range(length):
        float_comparison, diff = compare_floats(line_f1[j], line_f2[j])
        line_comparison = float_comparison & line_comparison
        if diff > max_diff:
            max_diff = diff

    if line_comparison:
        return True, max_diff

    return False, max_diff


def compare_files(
        file1: str,
        file2: str,
        nb_tests: int,
        precision: str,
) -> (bool, float):
    """Compare two files line wise using compare_lines."""
    f1 = Path.open(Path(file1))
    f2 = Path.open(Path(file2))

    max_diff_file = 0
    line_comparison = True

    for (line_f1, line_f2, line_counter) \
            in zip(f1, f2, range(nb_tests + 1), strict=False):
        float_comparison, max_diff_line = compare_lines(line_f1, line_f2, precision)
        line_comparison = float_comparison & line_comparison

        if max_diff_line > max_diff_file:
            max_diff_file = max_diff_line

        if line_counter == int(nb_tests):
            break

    f1.close()
    f2.close()

    if line_comparison:
        return True, max_diff_file

    return False, max_diff_file


def cli_compare(
        reference_file: str,
        c_file: str,
        nb_tests: int,
        precision: str,
) -> None:
    """Compare two files."""
    _, max_diff_file = compare_files(reference_file, c_file, nb_tests, precision)

    print(f"   Max absolute error for {nb_tests} test(s): {max_diff_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to verify the semantic preservation",
    )

    parser.add_argument(
        "reference_file",
        help="File with the inference output of the reference machine learning framework",
    )
    parser.add_argument(
        "c_file",
        help="File with the inference output of the studied machine learning framework",
    )
    parser.add_argument(
        "nb_tests",
        help="Number of inferences process to compare",
    )
    parser.add_argument(
        "--precision",
        help="Precision of the data studied. Default is float32",
    )

    args = parser.parse_args()

    precision = args.precision if args.precision else "float"

    cli_compare(args.reference_file, args.c_file, args.nb_tests, precision)
