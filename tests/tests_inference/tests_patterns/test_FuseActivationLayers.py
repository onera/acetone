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

import numpy as np
import onnx
import tensorflow as tf

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestFuseActivationsLayers(acetoneTestCase.AcetoneTestCase):
    """Test for FuseActivationsLayers pattern."""

    def testReLu(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testSigmoid(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testLeakyRelu(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testTanh(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testExp(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testLog(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testClip(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        min_value = np.random.rand(1)
        min_initializer = acetoneTestCase.create_initializer_tensor(name="min",
                                                                    tensor_array=min_value,
                                                                    data_type=onnx.TensorProto.FLOAT)

        max_value = min_value + np.random.rand(1)
        max_initializer = acetoneTestCase.create_initializer_tensor(name="max",
                                                                    tensor_array=max_value,
                                                                    data_type=onnx.TensorProto.FLOAT)

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
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor, min_initializer, max_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testActivationNotOnlyOutput(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        conv2_output_name = "output_conv2"
        conv2_in_channels = 5
        conv2_out_channels = 5
        conv2_kernel_shape = (7, 7)
        conv2_W = np.random.rand(
            conv2_out_channels, conv2_in_channels, *conv2_kernel_shape
        ).astype(np.float32)
        conv2_B = np.random.rand(conv2_out_channels).astype(np.float32)
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
            op_type="Conv",
            inputs=[
                conv1_output_name,
                conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name,
            ],
            outputs=[conv2_output_name],
            kernel_shape=conv2_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        relu_output_name = "output_relu"
        activation_node = onnx.helper.make_node(
            op_type="Relu",
            inputs=[conv1_output_name],
            outputs=[relu_output_name],
        )

        add_node = onnx.helper.make_node(
            op_type="Add",
            inputs=[conv2_output_name,relu_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node, activation_node, add_node],
            name="Conv",
            inputs=[X],
            outputs=[Y],
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

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])

    def testThreeSuccessiveActivationLayers(self):
        testshape = (1,3,10,10)
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_name = "output_conv1"
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
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        relu_output_name = "output_relu"
        activation_node1 = onnx.helper.make_node(
            op_type="Relu",
            inputs=[conv1_output_name],
            outputs=[relu_output_name],
        )

        tanh_output_name = "output_tanh"
        activation_node2 = onnx.helper.make_node(
            op_type="Tanh",
            inputs=[relu_output_name],
            outputs=[tanh_output_name],
        )

        sigmoid_output_name = "output_sigmoid"
        activation_node3 = onnx.helper.make_node(
            op_type="Sigmoid",
            inputs=[tanh_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, activation_node1, activation_node2, activation_node3],
            name="Conv",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
            verbose=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])
        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])


if __name__ == "__main__":
    acetoneTestCase.main()
