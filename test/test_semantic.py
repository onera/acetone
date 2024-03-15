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
import subprocess
import numpy as np
import keras
import unittest
import tensorflow as tf
import json
from keras.layers import Input, Dense

tf.keras.backend.set_floatx('float32')

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

def run_acetone_for_test(model:keras.Model, datatest_path:str=''):
    model.save('./tmp_dir/model.h5')

    cmd = 'python3'+' ../src/main.py '+'./tmp_dir/model.h5'+' inference'+' 1'+' std_gemm_nn'+' ./tmp_dir/acetone '+datatest_path
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        return "\nC code generation failed"
    
    cmd = 'make '+'-C'+' ./tmp_dir/acetone'+' all'
    print(cmd)
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        return "\nC code compilation failed"
    
    cmd = './tmp_dir/acetone/inference'+' ./tmp_dir/acetone/output_c.txt'
    result = subprocess.run(cmd.split(" ")).returncode
    if result != 0:
        return "\nC code inference failed"

    output = read_output('./tmp_dir/acetone/output_c.txt')

    subprocess.run(['rm','-r','tmp_dir/'])

    return output

class TestDenseLayer(unittest.TestCase):
    """Test for Dense Layer"""

    def test_Dense1(self):
        testshape = (5,23,16)
        units = 100

        input = Input(testshape)
        out = Dense(units, activation=None)(input)

        model = keras.Model(input,out)
        print(model.summary())
        dataset = create_dataset(testshape)


        acetone_result = run_acetone_for_test(model,'./temp_dir/dataset.txt')
        keras_result = model.predict(dataset)

        self.assertEqual(acetone_result,keras_result)

if __name__ == '__main__':
    unittest.main()