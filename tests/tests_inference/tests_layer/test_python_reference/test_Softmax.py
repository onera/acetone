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

acetoneTestCase_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense

import onnx
import onnxruntime as rt

tf.keras.backend.set_floatx('float32')


class TestSoftmax(acetoneTestCase.AcetoneTestCase):
    """Test for Softmax Layer"""
    
    def test_Softmax(self):
        testshape = (1,1,16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation='softmax', bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(list(acetone_result[1])), list(keras_result))
    
    def testSoftmaxONNX(self):
        testshape = (1,1, 1, 56)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,1, 1, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,1, 1, 56])

        activation_node = onnx.helper.make_node(
            op_type="Softmax",
            inputs=[model_input_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        sess = rt.InferenceSession(self.tmpdir_name+'/model.onnx')
        input_name = sess.get_inputs()[0].name
        result = sess.run(None,{input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx', self.tmpdir_name+'/dataset.txt')

        self.assertListAlmostEqual(list(list(acetone_result[1])), list(onnx_result))
    
    
if __name__ == '__main__':
    acetoneTestCase.main()