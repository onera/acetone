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
from keras.layers import Concatenate, Conv2D, Input

acetoneTestCase_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestConcate(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testConcatenateChannels(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        x2 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        out = Concatenate(axis=3)([x1, x2])

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testConcatenateHeights(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        x2 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        out = Concatenate(axis=1)([x1, x2])

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testConcatenateWidth(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        x2 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            bias_initializer="he_normal",
            padding="same",
            data_format="channels_last",
        )(input)
        out = Concatenate(axis=2)([x1, x2])

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[1]), list(keras_result))

    def testConcatChannelsONNX(self):
        model_input_name = "X"
        model_input_channels = 3
        model_input_height = 32
        model_input_width = 32
        testshape = (1, model_input_channels, model_input_height, model_input_width)
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, model_input_channels, model_input_height, model_input_width],
        )
        model_output_name = "Y"
        model_output_channels = 60
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, 2 * model_output_channels, model_input_height, model_input_width],
        )

        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = "Conv2_Y"
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Concat",
            inputs=[conv1_output_node_name, conv2_output_node_name],
            outputs=[model_output_name],
            axis=1,
        )

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node, conv2_node, merging_node],
            name="Concat",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testConcatHeightONNX(self):
        model_input_name = "X"
        model_input_channels = 3
        model_input_height = 32
        model_input_width = 32
        testshape = (1, model_input_channels, model_input_height, model_input_width)
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, model_input_channels, model_input_height, model_input_width],
        )
        model_output_name = "Y"
        model_output_channels = 60
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, model_output_channels, 2 * model_input_height, model_input_width],
        )

        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = "Conv2_Y"
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Concat",
            inputs=[conv1_output_node_name, conv2_output_node_name],
            outputs=[model_output_name],
            axis=2,
        )

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node, conv2_node, merging_node],
            name="Concat",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))

    def testConcatWidthONNX(self):
        model_input_name = "X"
        model_input_channels = 3
        model_input_height = 32
        model_input_width = 32
        testshape = (1, model_input_channels, model_input_height, model_input_width)
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [None, model_input_channels, model_input_height, model_input_width],
        )
        model_output_name = "Y"
        model_output_channels = 60
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [None, model_output_channels, model_input_height, 2 * model_input_width],
        )

        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = "Conv2_Y"
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(
            conv1_out_channels,
            conv1_in_channels,
            *conv1_kernel_shape,
        ).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT,
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Concat",
            inputs=[conv1_output_node_name, conv2_output_node_name],
            outputs=[model_output_name],
            axis=3,
        )

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node, conv2_node, merging_node],
            name="Concat",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[1]), list(onnx_result))


if __name__ == "__main__":
    acetoneTestCase.main()
