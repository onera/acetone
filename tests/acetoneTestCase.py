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

class AcetoneTestCase(unittest.TestCase):

    def assertListAlmostEqual(self, first, second, rtol=1e-07, atol=1e-07):
        return np.testing.assert_allclose(first, second, rtol=rtol, atol=atol)

def read_output(output_path:str):
    with open(output_path,'r') as f:
        line = f.readline()
        line = line[:-2].split(' ')
        line = list(map(float,line))
        line=np.array(line)
    f.close()
    return line

def create_dataset(shape):
    subprocess.run(['mkdir','tmp_dir/'])
    dataset = np.float32(np.random.default_rng(seed=10).random((1,)+ shape))
    with open('./tmp_dir/dataset.txt', 'w') as filehandle:
        for i in range(dataset.shape[0]):
            row = (dataset[i]).flatten(order='C')
            json.dump(row.tolist(), filehandle)
            filehandle.write('\n')
    filehandle.close()
    return dataset

def run_acetone_for_test(model:str, datatest_path:str='',conv_algo:str='std_gemm_nn'):
 
    cmd = 'python3'+' src/cli_acetone.py '+model+' inference'+' 1 '+conv_algo+' ./tmp_dir/acetone '+datatest_path
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        print("\nC code generation failed")
        subprocess.run(['rm','-r','tmp_dir/'])
        return np.array([])
    
    cmd = 'make '+'-C'+' ./tmp_dir/acetone'+' all'
    print(cmd)
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        print("\nC code compilation failed")
        subprocess.run(['rm','-r','tmp_dir/'])
        return np.array([])
    
    cmd = './tmp_dir/acetone/inference'+' ./tmp_dir/acetone/output_c.txt'
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        print("\nC code inference failed")
        subprocess.run(['rm','-r','tmp_dir/'])
        return np.array([])
    
    output = read_output('./tmp_dir/acetone/output_c.txt')
    subprocess.run(['rm','-r','tmp_dir/'])

    return output

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