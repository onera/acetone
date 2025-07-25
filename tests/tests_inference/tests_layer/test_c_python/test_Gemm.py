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

from tests.tests_inference import acetoneTestCase


class TestGemm(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testGemm_nn(self):
        testshape = (3, 5)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [3, 5],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [3, 7],
        )

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.arange(35).reshape((5, 7)).astype(np.float32)
        Gemm_W_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_W_name,
            tensor_array=Gemm_W,
            data_type=onnx.TensorProto.FLOAT,
        )

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(7).astype(np.float32)
        Gemm_B_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_B_name,
            tensor_array=Gemm_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm",
            op_type="Gemm",
            inputs=[model_input_name, Gemm_W_name, Gemm_B_name],
            outputs=[model_output_name],
            alpha=2.0,
            beta=2.0,
            transA=0,
            transB=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[gemm_node],
            name="Gemm",
            inputs=[X],
            outputs=[Y],
            initializer=[Gemm_W_initializer, Gemm_B_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGemm_nt(self):
        testshape = (3, 5)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [3, 5],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [3, 7],
        )

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(7, 5).astype(np.float32)
        Gemm_W_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_W_name,
            tensor_array=Gemm_W,
            data_type=onnx.TensorProto.FLOAT,
        )

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(7).astype(np.float32)
        Gemm_B_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_B_name, tensor_array=Gemm_B, data_type=onnx.TensorProto.FLOAT,
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm",
            op_type="Gemm",
            inputs=[model_input_name, Gemm_W_name, Gemm_B_name],
            outputs=[model_output_name],
            alpha=2.0,
            beta=2.0,
            transA=0,
            transB=1,
        )

        graph = onnx.helper.make_graph(
            nodes=[gemm_node],
            name="Gemm",
            inputs=[X],
            outputs=[Y],
            initializer=[Gemm_W_initializer, Gemm_B_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGemm_tn(self):
        testshape = (5, 3)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [5, 3],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [3, 7],
        )

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(5, 7).astype(np.float32)
        Gemm_W_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_W_name,
            tensor_array=Gemm_W,
            data_type=onnx.TensorProto.FLOAT,
        )

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(7).astype(np.float32)
        Gemm_B_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_B_name,
            tensor_array=Gemm_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm",
            op_type="Gemm",
            inputs=[model_input_name, Gemm_W_name, Gemm_B_name],
            outputs=[model_output_name],
            alpha=2.0,
            beta=2.0,
            transA=1,
            transB=0,
        )

        graph = onnx.helper.make_graph(
            nodes=[gemm_node],
            name="Gemm",
            inputs=[X],
            outputs=[Y],
            initializer=[Gemm_W_initializer, Gemm_B_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGemm_tt(self):
        testshape = (5, 3)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name,
            onnx.TensorProto.FLOAT,
            [5, 3],
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [3, 7],
        )

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(7, 5).astype(np.float32)
        Gemm_W_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_W_name,
            tensor_array=Gemm_W,
            data_type=onnx.TensorProto.FLOAT,
        )

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(7).astype(np.float32)
        Gemm_B_initializer = acetoneTestCase.create_initializer_tensor(
            name=Gemm_B_name,
            tensor_array=Gemm_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm",
            op_type="Gemm",
            inputs=[model_input_name, Gemm_W_name, Gemm_B_name],
            outputs=[model_output_name],
            alpha=2.0,
            beta=2.0,
            transA=1,
            transB=1,
        )

        graph = onnx.helper.make_graph(
            nodes=[gemm_node],
            name="Gemm",
            inputs=[X],
            outputs=[Y],
            initializer=[Gemm_W_initializer, Gemm_B_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
        )

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))


if __name__ == "__main__":
    acetoneTestCase.main()
