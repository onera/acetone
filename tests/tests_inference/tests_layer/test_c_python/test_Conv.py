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
import tensorflow as tf
from keras import layers

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestConv(acetoneTestCase.AcetoneTestCase):
    """Test for Conv Layer"""

    def testConv_6loops(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = layers.Input(testshape)
        out = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                            padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              conv_algo="6loops")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testConv_indirect_gemm_nn(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = layers.Input(testshape)
        out = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                            padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              conv_algo="indirect_gemm_nn")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testConv_std_gemm_nn(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = layers.Input(testshape)
        out = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                            padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              conv_algo="std_gemm_nn")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testConv_std_gemm_nnONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 200, 200])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 200, 200])
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
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
                conv1_B_initializer_tensor_name,
            ],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testConv_6loopsONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 200, 200])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 200, 200])
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
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
                conv1_B_initializer_tensor_name,
            ],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              conv_algo="6loops")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testConv_indirect_gemm_nnONNX(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 200, 200])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 200, 200])
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
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
                conv1_B_initializer_tensor_name,
            ],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              conv_algo="indirect_gemm_nn")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))


if __name__ == "__main__":
    acetoneTestCase.main()
