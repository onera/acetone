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
import unittest
import numpy as np

acetoneTestCase_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase
import keras
import onnx
import onnxruntime as rt
from keras.layers import Input, ZeroPadding2D

from tests.tests_inference import acetoneTestCase


class TestPad(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def test_Pads(self):
        testshape = (10, 10, 3)

        input = Input(testshape)
        out = ZeroPadding2D(padding=(1, 1))(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    @unittest.skip("Not yet implemented")
    def test_edge_pad(self):
        testshape = (1, 5, 20, 20)

        pads_name = "pads"
        pads = np.random.randint(10, size=8)
        pads[0] = 0
        pads[4] = 0
        pads_initializer = acetoneTestCase.create_initializer_tensor(
            pads_name,
            pads,
            onnx.TensorProto.INT64,
        )

        cst_name = "constant"
        cst = np.random.rand(1)
        cst_initializer = acetoneTestCase.create_initializer_tensor(
            cst_name,
            cst,
            onnx.TensorProto.FLOAT,
        )

        out_shape = (
            1,
            int(testshape[1] + pads[1] + pads[5]),
            int(testshape[2] + pads[2] + pads[6]),
            int(testshape[3] + pads[3] + pads[7]),
        )

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, testshape
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, out_shape
        )

        activation_node = onnx.helper.make_node(
            op_type="Pad",
            inputs=[model_input_name, pads_name, cst_name],
            outputs=[model_output_name],
            mode="edge"
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="edge_pad",
            inputs=[X],
            outputs=[Y],
            initializer=[pads_initializer, cst_initializer],
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
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    @unittest.skip("Not yet implemented")
    def test_wrap_pad(self):
        testshape = (1, 5, 20, 20)

        pads_name = "pads"
        pads = np.random.randint(10, size=8)
        pads[0] = 0
        pads[4] = 0
        pads_initializer = acetoneTestCase.create_initializer_tensor(
            pads_name,
            pads,
            onnx.TensorProto.INT64,
        )

        cst_name = "constant"
        cst = np.random.rand(1)
        cst_initializer = acetoneTestCase.create_initializer_tensor(
            cst_name,
            cst,
            onnx.TensorProto.FLOAT,
        )

        out_shape = (
            1,
            int(testshape[1] + pads[1] + pads[5]),
            int(testshape[2] + pads[2] + pads[6]),
            int(testshape[3] + pads[3] + pads[7]),
        )

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, testshape
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, out_shape
        )

        activation_node = onnx.helper.make_node(
            op_type="Pad",
            inputs=[model_input_name, pads_name, cst_name],
            outputs=[model_output_name],
            mode="wrap"
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="wrap_pad",
            inputs=[X],
            outputs=[Y],
            initializer=[pads_initializer, cst_initializer],
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
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def test_constant_pad(self):
        testshape = (1, 5, 20, 20)

        pads_name = "pads"
        pads = np.random.randint(10, size=8)
        pads[0] = 0
        pads[4] = 0
        pads_initializer = acetoneTestCase.create_initializer_tensor(
            pads_name,
            pads,
            onnx.TensorProto.INT64,
        )

        cst_name = "constant"
        cst = np.random.rand()
        cst_initializer = acetoneTestCase.create_initializer_tensor(
            cst_name,
            np.array(cst),
            onnx.TensorProto.FLOAT,
        )

        out_shape = (
            1,
            int(testshape[1] + pads[1] + pads[5]),
            int(testshape[2] + pads[2] + pads[6]),
            int(testshape[3] + pads[3] + pads[7]),
        )

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, testshape
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, out_shape
        )

        activation_node = onnx.helper.make_node(
            op_type="Pad",
            inputs=[model_input_name, pads_name, cst_name],
            outputs=[model_output_name],
            mode="constant"
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="constant_pad",
            inputs=[X],
            outputs=[Y],
            initializer=[pads_initializer, cst_initializer],
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
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    @unittest.skip("Not yet implemented")
    def test_reflect_pad(self):
        testshape = (1, 5, 20, 20)

        pads_name = "pads"
        pads = np.random.randint(10, size=8)
        pads[0] = 0
        pads[4] = 0
        pads_initializer = acetoneTestCase.create_initializer_tensor(
            pads_name,
            pads,
            onnx.TensorProto.INT64,
        )

        cst_name = "constant"
        cst = np.random.rand(1)
        cst_initializer = acetoneTestCase.create_initializer_tensor(
            cst_name,
            cst,
            onnx.TensorProto.FLOAT,
        )

        out_shape = (
            1,
            int(testshape[1] + pads[1] + pads[5]),
            int(testshape[2] + pads[2] + pads[6]),
            int(testshape[3] + pads[3] + pads[7]),
        )

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, testshape
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, out_shape
        )

        activation_node = onnx.helper.make_node(
            op_type="Pad",
            inputs=[model_input_name, pads_name, cst_name],
            outputs=[model_output_name],
            mode="reflect"
        )

        graph = onnx.helper.make_graph(
            nodes=[activation_node],
            name="reflect_pad",
            inputs=[X],
            outputs=[Y],
            initializer=[pads_initializer, cst_initializer],
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
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

if __name__ == "__main__":
    acetoneTestCase.main()
