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


class TestTranspose(acetoneTestCase.AcetoneTestCase):
    """Test for Transpose Layer"""

    def testTransposeONNX1(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 1, 2, 3))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])

    def testTransposeONNX2(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 1, 3, 2))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])

    def testTransposeONNX3(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 2, 1, 3))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])

    def testTransposeONNX4(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 2, 3, 1))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])

    def testTransposeONNX5(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 3, 1, 2))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])

    def testTransposeONNX6(self):
        testshape = (1, 5, 64, 56)

        perm = np.array((0, 3, 2, 1))

        out_shape = [testshape[i] for i in perm[1:]]
        out_shape = [None] + out_shape

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 64, 56])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        activation_node = onnx.helper.make_node(
            op_type="Transpose",
            inputs=[model_input_name],
            outputs=[model_output_name],
            perm=perm,
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="transpose",
            inputs=[X],
            outputs=[Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])


if __name__ == "__main__":
    acetoneTestCase.main()
