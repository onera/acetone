"""*******************************************************************************
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

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from acetone_nnet.cli.generate import cli_acetone


class AcetoneTestCase(unittest.TestCase):
    """TestCase class for inference tests."""

    def _redirect_generated_code(self, target: str | Path) -> None:
        """Redirect test output to the selected folder.

        Used to check the generated code for a failing test.
        """
        from shutil import rmtree

        target = Path(target)
        if target.exists():
            rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        self.tmpdir_name = str(target)

    def setUp(self) -> None:
        """Create a temp_dir."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_name = self.tmpdir.name

    def tearDown(self) -> None:
        """Destroy a temp_dir."""
        self.tmpdir.cleanup()

    def assertListAlmostEqual(
        self,
        first: np.ndarray,
        second: np.ndarray,
        rtol: float = 5e-06,
        atol: float = 5e-06,
        err_msg: str = "",
    ) -> None:
        return np.testing.assert_allclose(
            first,
            second,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )


def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
    data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
) -> onnx.TensorProto:
    """Create a TensorProto."""
    return onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )


def read_output_c(path_to_output: str, target: str, trimline: int = -2) -> np.ndarray:
    """Read the output file of the C code."""
    with open(path_to_output) as f:
        line = f.readline()
        words = line[:trimline].split(" ")
        if target == "generic":
            return np.array(list(map(float.fromhex, words)))
        return np.array(list(map(int, words)))


def read_output_python(path_to_output: str, target: str) -> np.ndarray:
    return read_output_c(path_to_output, target, trimline=-3)


def create_dataset(tmpdir: str, shape: tuple):
    """Create a random dataset."""
    dataset = np.float32(np.random.default_rng(seed=10).random((1, *shape)))
    with open(tmpdir + "/dataset.txt", "w") as filehandle:
        row = (dataset[0]).flatten(order="C")
        row = [float(f).hex().replace("0000000p", "p") for f in row]
        json.dump(row, filehandle)
        filehandle.write("\n")
    filehandle.close()
    return dataset


def run_acetone_for_test(
    tmpdir_name: str,
    model: str | Path,
    datatest_path: str | np.ndarray | None = None,
    conv_algo: str = "std_gemm_nn",
    normalize=False,
    optimization: bool = False,
    verbose:bool = True,
    run_generated=True,
    run_reference=True,
    target="generic",
    bin_dataset=False,
    gen_data_format="channels_first"
):
    cli_acetone(
        model_file=model,
        function_name="inference",
        nb_tests=1,
        conv_algorithm=conv_algo,
        output_dir=tmpdir_name,
        test_dataset_file=datatest_path,
        normalize=normalize,
        verbose=verbose,
        optimization=optimization,
        target=target,
        bin_dataset=bin_dataset,
        gen_data_format=gen_data_format
    )

    if run_reference:
        output_python = read_output_python(
            tmpdir_name + "/output_python.txt",
            target,
        ).flatten()
    else:
        output_python = None

    if run_generated:
        cmd = ["make", "-C", tmpdir_name, "all"]
        result = subprocess.run(cmd, check=False).returncode
        if result != 0:
            print(f"\nC code compilation failed code {result}")
            return np.array([]), output_python

        cmd = [tmpdir_name + "/inference", tmpdir_name + "/output_c.txt"]
        result = subprocess.run(cmd, check=False).returncode
        if result != 0:
            print(f"\nC code inference failed code {result}")
            # try to read file anyways ? return np.array([]), output_python

        output_c = read_output_c(tmpdir_name + "/output_c.txt", target).flatten()
    else:
        output_c = None

    return output_c, output_python
