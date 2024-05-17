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
from keras.layers import Input, Dense, Conv2D

import onnx 

tf.keras.backend.set_floatx('float32')


class TestActivation(acetoneTestCase.AcetoneTestCase):
    """Test for Activations Layer"""
    
    def testReLu(self):
        testshape = (1,1,16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation='relu', bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testLeakyReLu(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='leaky_relu', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testSigmoid(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='sigmoid', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testTanh(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='tanh', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testReLuONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Relu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testSigmoidONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Sigmoid",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testLeakyReludONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="LeakyRelu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
            alpha = np.random.random()/10
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testTanhONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Tanh",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testExpONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Exp",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
    
    def testLogONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Log",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testClipONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        min_initializer = acetoneTestCase.create_initializer_tensor(name='min',
                                                                    tensor_array=np.random.rand(1)*10,
                                                                    data_type=onnx.TensorProto.FLOAT)

        max_initializer = acetoneTestCase.create_initializer_tensor(name='max',
                                                                    tensor_array=np.random.rand(1)*20,
                                                                    data_type=onnx.TensorProto.FLOAT)

        activation_node = onnx.helper.make_node(
            op_type="Clip",
            inputs=[conv1_output_name,'min','max'],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor,min_initializer,max_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == '__main__':
    acetoneTestCase.main()