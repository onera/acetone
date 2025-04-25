"""Entry point for the acetone_compare command line tool.

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


def compare_floats(
    a: float,
    b: float,
    epsilon: float = (128 * sys.float_info.epsilon),
    abs_th: float = sys.float_info.min,
) -> tuple[bool, float, float]:
    """Compare a-b to abs_th."""
    diff = abs(a - b)

    if a == b:
        return True, diff, 0.0

    norm = min((abs(a) + abs(b)), sys.float_info.max)
    rel_diff = diff / (norm / 2) if norm != 0 else 0.0
    if diff < max(abs_th, epsilon * norm):
        return True, diff, rel_diff

    return False, diff, rel_diff


def preprocess_line(line: str, precision: str) -> list:
    """Cast a list of str to a list of float."""
    values: list = [float.fromhex(x) for x in line.split(" ") if x.strip()]
    if precision == "double":
        values = list(map(np.float64, values))
    elif precision == "float":
        values = list(map(np.float32, values))
    return values


def compare_lines(
        line_f1: str,
        line_f2: str,
        precision: str,
) -> tuple[bool, float, float]:
    """Compare two lines element wise using compare_floats."""
    values_f1 = preprocess_line(line_f1, precision)
    values_f2 = preprocess_line(line_f2, precision)
    max_diff = 0.0
    max_rel_diff = 0.0
    length = len(values_f1)
    line_comparison = True

    for j in range(length):
        float_comparison, diff, rel_diff = compare_floats(values_f1[j], values_f2[j])
        line_comparison = float_comparison & line_comparison
        max_diff = max(diff, max_diff)
        max_rel_diff = max(rel_diff, max_rel_diff)

    if line_comparison:
        return True, max_diff, max_rel_diff

    return False, max_diff, max_rel_diff


def compare_files(
    file1: str,
    file2: str,
    nb_tests: int,
    precision: str,
) -> tuple[bool, float, float]:
    """Compare two files line wise using compare_lines."""
    f1 = Path.open(Path(file1))
    f2 = Path.open(Path(file2))

    max_diff_file = 0.0
    max_rel_diff_file = 0.0
    line_comparison = True

    for line_f1, line_f2, line_counter in zip(
        f1,
        f2,
        range(nb_tests + 1),
        strict=False,
    ):
        float_comparison, max_diff_line, max_rel_diff_line = (
            compare_lines(line_f1, line_f2, precision)
        )
        line_comparison = float_comparison & line_comparison

        max_diff_file = max(max_diff_line, max_diff_file)
        max_rel_diff_file = max(max_rel_diff_line, max_rel_diff_file)

        if line_counter == int(nb_tests):
            break

    f1.close()
    f2.close()

    if line_comparison:
        return True, max_diff_file, max_rel_diff_file

    return False, max_diff_file, max_rel_diff_file


def cli_compare(
    reference_file: str,
    c_file: str,
    nb_tests: int,
    precision: str = "float",
) -> None:
    """Compare two files."""
    _, max_diff_file, max_rel_diff_file = (
        compare_files(reference_file, c_file, nb_tests, precision)
    )

    print(f"    Max absolute error for {nb_tests} test(s): {max_diff_file}")
    print(f"    Max relative error for {nb_tests} test(s): {max_rel_diff_file}\n")


def acetone_compare() -> None:
    """Compare two text file containing inference output(s)."""
    parser = argparse.ArgumentParser(
        description=acetone_compare.__doc__,
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
        type=int,
    )
    parser.add_argument(
        "--precision",
        help="Precision of the data studied. Default is float32",
    )

    args = parser.parse_args()

    precision = args.precision if args.precision else "float"

    cli_compare(
        reference_file=args.reference_file,
        c_file=args.c_file,
        nb_tests=args.nb_tests,
        precision=precision,
    )


if __name__ == "__main__":
    acetone_compare()