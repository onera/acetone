"""Debug tools for ACETONE.

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

from pathlib import Path

import numpy as np


def compare_result(
        acetone_result: list | np.ndarray,
        reference_result: list | np.ndarray,
        targets: list,
        verbose: bool = False,
        atol: float = 5e-06,
        rtol: float = 5e-06
) -> bool:
    """Compare two list and return of they are similar."""
    if len(acetone_result) != len(reference_result):
        msg = "Error: both result don't have the same size"

        if verbose:
            print(msg)

        return False

    correct = True
    count = 0
    first = None
    for i in range(len(acetone_result)):
        print("--------------------------------------------")
        print("Comparing", targets[i])
        try:
            np.testing.assert_allclose(
                acetone_result[i],
                reference_result[i],
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as msg:
            print("Error: output value of", targets[i], "incorrect")
            correct = False
            count += 1
            if not first:
                first = targets[i]
            if verbose:
                print(msg)
        print("--------------------------------------------")

    if verbose:
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Total number of error: {count}/{len(acetone_result)} ({count/len(acetone_result)*100:.3}%)")
        print(f"First error at {first}")
        print("++++++++++++++++++++++++++++++++++++++++++++")

    return correct

def reorder_outputs(
        outputs: list[np.ndarray],
        targets: list[str],
) -> (list[np.ndarray], list[str]):
    """Rearrange the to match the reference."""
    ordered_outputs = []
    ordered_targets = []

    dictionary = {}
    for i in range(len(targets)):
        dictionary[int(targets[i].split(" ")[-1])] = (
            outputs[i], targets[i],
        )

    for element in sorted(dictionary.items(), key=lambda item: item[0]):
        ordered_outputs.append(element[1][0])
        ordered_targets.append(element[1][1])

    return ordered_outputs, ordered_targets

def extract_outputs_c(
        path_to_output: str | Path,
        data_type: str,
        nb_targets: int,
) -> (list[np.ndarray], list[str]):
    """Get the outputs values from the debug_file."""
    list_result = []
    targets = []
    to_transpose = False
    with Path.open(Path(path_to_output)) as f:
        for i, line in enumerate(f):

            if i % 2 == 0:
                line = line[:-1].split(" ")
                targets.append(line[0] + " " + line[1])
                to_transpose = int(line[2])
                if to_transpose:
                    shape = (int(line[3]), int(line[4]), int(line[5]))
            else:
                line = line[:-2].split(" ")
                if data_type == "int":
                    line = list(map(int, line))
                elif data_type == "double":
                    line = list(map(float, line))
                elif data_type == "float":
                    line = list(map(np.float32, line))
                line = np.array(line)

                if to_transpose:
                    line = np.reshape(line, shape)
                    line = np.transpose(line, (1, 2, 0))
                    line = line.flatten()

                list_result.append(line)

            if i == 2 * nb_targets - 1:
                break
    f.close()

    return reorder_outputs(list_result, targets)

def extract_outputs_python(
        result_python: tuple[list],
        targets_indices: list[str],
) -> (list[np.ndarray], list[str]):
    """Get the outputs values from the output_py file."""
    outputs = []
    targets = []
    for indices in targets_indices:
        for i in range(len(result_python[1])):
            if indices == int(result_python[1][i][-1]):
                outputs.append(result_python[0][i])
                targets.append(result_python[1][i])

    return reorder_outputs(outputs, targets)
