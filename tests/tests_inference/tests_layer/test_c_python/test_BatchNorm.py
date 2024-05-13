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
from keras.layers import Input, BatchNormalization, Conv2D

import onnx

tf.keras.backend.set_floatx('float32')


class TestBatchNormalization(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testBatchNorm(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = BatchNormalization(axis=-1, gamma_initializer='he_normal', beta_initializer='he_normal', moving_mean_initializer='he_normal',moving_variance_initializer='ones')(x1)
        
        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')

        self.assertListAlmostEqual(list(acetone_result[1]), list(acetone_result[0]))
    
    def testBatchNorm2(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_node_name = "Conv1_Y"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
        conv1_pads = (3,3,3,3)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT)
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT)

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads = conv1_pads,
            strides = (1,1),
        )

        scale_name = 'scale'
        scale = np.random.rand(5)
        scale_initializer = acetoneTestCase.create_initializer_tensor(name = scale_name,
                                                                       tensor_array = scale,
                                                                       data_type = onnx.TensorProto.FLOAT)
        
        bias_name = 'bias'
        bias = np.random.rand(5)
        bias_initializer = acetoneTestCase.create_initializer_tensor(name = bias_name,
                                                                       tensor_array = bias,
                                                                       data_type = onnx.TensorProto.FLOAT)
        
        mean_name = 'mean'
        mean = np.random.rand(5)
        mean_initializer = acetoneTestCase.create_initializer_tensor(name = mean_name,
                                                                       tensor_array = mean,
                                                                       data_type = onnx.TensorProto.FLOAT)
        
        var_name = 'var'
        var = np.random.rand(5)
        var_initializer = acetoneTestCase.create_initializer_tensor(name = var_name,
                                                                       tensor_array = var,
                                                                       data_type = onnx.TensorProto.FLOAT)
        
        gather_node = onnx.helper.make_node(
            name = 'BatchNormalization',
            op_type = 'BatchNormalization',
            inputs = [conv1_output_node_name,scale_name,bias_name,mean_name,var_name],
            outputs = [model_output_name],
            epsilon = 1e-05,
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,gather_node],
            name = 'BatchNormalization',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor,scale_initializer,bias_initializer,mean_initializer,var_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
if __name__ == '__main__':
    acetoneTestCase.main()