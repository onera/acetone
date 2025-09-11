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

from tests.tests_inference import acetoneTestCase


class TestGatherElements(acetoneTestCase.AcetoneTestCase):
    """Test for GatherElements Layer"""

    def testGatherElements1(self):
        in_shape = list(np.random.randint(1, 100, size=3))
        in_shape = (1, int(in_shape[0]), int(in_shape[1]), int(in_shape[2]))
        out_shape = (1, int(np.random.randint(1, high=1+in_shape[1])), int(in_shape[2]), int(in_shape[3]))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               in_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        indices_name = "indice"
        indices = np.random.randint(0, high=in_shape[1], size=out_shape)
        indices_initializer = acetoneTestCase.create_initializer_tensor(name=indices_name,
                                                                        tensor_array=indices,
                                                                        data_type=onnx.TensorProto.INT32)

        GatherElements_node = onnx.helper.make_node(
            name="GatherElements",
            op_type="GatherElements",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=1,
        )
        graph = onnx.helper.make_graph(
            nodes=[GatherElements_node],
            name="GatherElements",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, in_shape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[0], onnx_result)

    def testGatherElements2(self):
        in_shape = list(np.random.randint(1, 100, size=3))
        in_shape = (1, int(in_shape[0]), int(in_shape[1]), int(in_shape[2]))
        out_shape = (1, int(in_shape[1]), int(np.random.randint(1, 1+in_shape[2])), int(in_shape[3]))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               in_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        indices_name = "indice"
        indices = np.random.randint(in_shape[2], size=out_shape)
        indices_initializer = acetoneTestCase.create_initializer_tensor(name=indices_name,
                                                                        tensor_array=indices,
                                                                        data_type=onnx.TensorProto.INT32)

        GatherElements_node = onnx.helper.make_node(
            name="GatherElements",
            op_type="GatherElements",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=2,
        )
        graph = onnx.helper.make_graph(
            nodes=[GatherElements_node],
            name="GatherElements",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, in_shape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[0], onnx_result)

    def testGatherElements3(self):
        in_shape = list(np.random.randint(1, 100, size=3))
        in_shape = (1, int(in_shape[0]), int(in_shape[1]), int(in_shape[2]))
        out_shape = (1, int(in_shape[1]), int(in_shape[2]), int(np.random.randint(1, 1+in_shape[3])))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               in_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        indices_name = "indice"
        indices = np.random.randint(in_shape[3], size=out_shape)
        indices_initializer = acetoneTestCase.create_initializer_tensor(name=indices_name,
                                                                        tensor_array=indices,
                                                                        data_type=onnx.TensorProto.INT32)

        GatherElements_node = onnx.helper.make_node(
            name="GatherElements",
            op_type="GatherElements",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=-1,
        )
        graph = onnx.helper.make_graph(
            nodes=[GatherElements_node],
            name="GatherElements",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, in_shape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[0], onnx_result)

    def testGatherElements2DINputs(self):
        in_shape = list(np.random.randint(1, 100, size=2))
        in_shape = (int(in_shape[0]), int(in_shape[1]))
        out_shape = (int(np.random.randint(1, high=1+in_shape[0])), int(in_shape[1]))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               in_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               out_shape)

        indices_name = "indice"
        indices = np.random.randint(0,high=in_shape[0], size=out_shape)
        indices_initializer = acetoneTestCase.create_initializer_tensor(name=indices_name,
                                                                        tensor_array=indices,
                                                                        data_type=onnx.TensorProto.INT32)

        GatherElements_node = onnx.helper.make_node(
            name="GatherElements",
            op_type="GatherElements",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=0,
        )
        graph = onnx.helper.make_graph(
            nodes=[GatherElements_node],
            name="GatherElements",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, in_shape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        sess = rt.InferenceSession(self.tmpdir_name + "/model.onnx")
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")
        self.assertListAlmostEqual(acetone_result[0], onnx_result)

if __name__ == "__main__":
    acetoneTestCase.main()
