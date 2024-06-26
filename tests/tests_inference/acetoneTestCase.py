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

import numpy as np
import unittest
import subprocess
import json
import onnx
import tempfile

import acetone_nnet

class AcetoneTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_name = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def assertListAlmostEqual(self, first:np.ndarray, second:np.ndarray, rtol=5e-06, atol=5e-06, err_msg = ''):
        return np.testing.assert_allclose(first, second, rtol=rtol, atol=atol, err_msg=err_msg)

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def read_output_c(path_to_output:str):
    with open(path_to_output,'r') as f:
        line = f.readline()
        line = line[:-2].split(' ')
        line = list(map(float,line))
        line=np.array(line)
    f.close()
    return line

def read_output_python(path_to_output:str):
    with open(path_to_output,'r') as f:
        line = f.readline()
        line = line[:-3].split(' ')
        line = list(map(float,line))
        line=np.array(line)
    f.close()
    return line


def create_dataset(tmpdir:str, shape:tuple):
    dataset = np.float32(np.random.default_rng(seed=10).random((1,)+ shape))
    with open(tmpdir+'/dataset.txt', 'w') as filehandle:
        row = (dataset[0]).flatten(order='C')
        json.dump(row.tolist(), filehandle)
        filehandle.write('\n')
    filehandle.close()
    return dataset

def run_acetone_for_test(tmpdir_name: str, model:str, datatest_path:str|None=None, conv_algo:str='std_gemm_nn', normalize=False):
    acetone_nnet.cli_acetone(model_file=model, function_name='inference', nb_tests=1, conv_algorithm=conv_algo, output_dir=tmpdir_name, test_dataset_file=datatest_path, normalize=normalize)
    output_python = read_output_python(tmpdir_name+'/output_python.txt')

    cmd = ['make', '-C', tmpdir_name, 'all']
    result = subprocess.run(cmd).returncode
    if result != 0:
        print("\nC code compilation failed")
        return np.array([]), output_python.flatten()
    
    cmd = [tmpdir_name+'/inference', tmpdir_name+'/output_c.txt']
    result = subprocess.run(cmd).returncode
    if result != 0:
        print("\nC code inference failed")
        return np.array([]), output_python.flatten()
    
    output_c = read_output_c(tmpdir_name+'/output_c.txt')
    
    return output_c.flatten(), output_python.flatten()