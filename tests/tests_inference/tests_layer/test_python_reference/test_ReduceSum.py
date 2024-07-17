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
import onnxruntime as rt

acetoneTestCase_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase


class TestReduceSum(acetoneTestCase.AcetoneTestCase):
    """Test for ReduceSum Layer"""

    def testReduceSum_c(self):
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
            [None, 1, 10, 10],
        )

        axes_name = "axes"
        axes = np.array([1])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_h(self):
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
            [None, 3, 1, 10],
        )

        axes_name = "axes"
        axes = np.array([2])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_w(self):
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
            [None, 3, 10, 1],
        )

        axes_name = "axes"
        axes = np.array([3])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_ch(self):
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
            [None, 1, 1, 10],
        )

        axes_name = "axes"
        axes = np.array([1, 2])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_cw(self):
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
            [None, 1, 10, 1],
        )

        axes_name = "axes"
        axes = np.array([1, 3])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_hw(self):
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
            [None, 3, 1, 1],
        )

        axes_name = "axes"
        axes = np.array([2, 3])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_chw(self):
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
            [None, 1, 1, 1],
        )

        axes_name = "axes"
        axes = np.array([1, 2, 3])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_empty_noop_false(self):
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
            [None, 1, 1, 1],
        )

        axes_name = "axes"
        axes = np.array([])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)

    def testReduceSum_empty_noop_True(self):
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
            [None, 3, 10, 10],
        )

        axes_name = "axes"
        axes = np.array([])
        axes_initializer = acetoneTestCase.create_initializer_tensor(
            name=axes_name,
            tensor_array=axes,
            data_type=onnx.TensorProto.INT64,
        )

        ReduceSum_node = onnx.helper.make_node(
            name="ReduceSum",
            op_type="ReduceSum",
            inputs=[model_input_name, axes_name],
            outputs=[model_output_name],
            keepdims=1,
            noop_with_empty_axes=1,
        )

        graph = onnx.helper.make_graph(
            nodes=[ReduceSum_node],
            name="ReduceSum",
            inputs=[X],
            outputs=[Y],
            initializer=[axes_initializer],
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
        )
        self.assertListAlmostEqual(acetone_result[1], onnx_result)


if __name__ == "__main__":
    acetoneTestCase.main()
