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
from keras.layers import Input, Conv2D, Add, Multiply, Subtract, Average, Maximum, Minimum

import onnx
import numpy as np
import onnxruntime as rt

tf.keras.backend.set_floatx('float32')


class TestBroadcast(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testAdd(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Add()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

    def testMul(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Multiply()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testSub(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Subtract()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testAvg(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Average()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testMax(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Maximum()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

    def testMin(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Minimum()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

    def testSubONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Sub",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Sub",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))
    
    def testMulONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Mul",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Mul",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))
    
    def testDivONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Div",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Div",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))
    
    def testMaxONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Max",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Max",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))
    
    def testMinONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Min",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Min",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))

    def testAddONNX(self):
        testshape = (1,3,32,32)
        model_input_name = "X"
        model_input_channels = 3
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =3
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Add",
            inputs=[model_input_name,model_input_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[merging_node],
            name="Add",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
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

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))

if __name__ == '__main__':
    acetoneTestCase.main()