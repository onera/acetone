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

import keras
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
from keras.layers import Conv2D, Dense, Input

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestActivation(acetoneTestCase.AcetoneTestCase):
    """Test for Activations Layer"""

    def testReLu(self):
        testshape = (1, 1, 16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation="relu", bias_initializer="he_normal")(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testLeakyReLu(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="leaky_relu",
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testSigmoid(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="sigmoid",
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testTanh(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="tanh",
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testReLuONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Relu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testSigmoidONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Sigmoid",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testLeakyReludONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="LeakyRelu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
            alpha=np.random.random() / 10,
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testTanhONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Tanh",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testExpONNX(self):
        rng = np.random.default_rng(42)
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = rng.random(
            (conv1_out_channels, conv1_in_channels) + tuple(conv1_kernel_shape),
            np.float32,
        )
        conv1_B = rng.random(conv1_out_channels, np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Exp",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        for i, (r, s) in enumerate(zip(acetone_result[1], onnx_result, strict=False)):
            if abs(r - s) > 5e-06:
                print(i, r, s, (r - s) / r)
        self.assertListAlmostEqual(
            list(acetone_result[1]),
            list(onnx_result),
            rtol=1e-05,
            atol=0.0,
        )

    def testLogONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Log",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testClipONNX(self):
        testshape = (1, 3, 10, 10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, 3, 10, 10],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 5, 10, 10],
        )

        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name,
                conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        min_value = np.random.rand(1)
        min_initializer = acetoneTestCase.create_initializer_tensor(
            name="min", tensor_array=min_value, data_type=onnx.TensorProto.FLOAT
        )

        max_value = min_value + np.random.rand(1)
        max_initializer = acetoneTestCase.create_initializer_tensor(
            name="max", tensor_array=max_value, data_type=onnx.TensorProto.FLOAT
        )

        activation_node = onnx.helper.make_node(
            op_type="Clip",
            inputs=[conv1_output_name, "min", "max"],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                min_initializer,
                max_initializer,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))


if __name__ == "__main__":
    acetoneTestCase.main()
