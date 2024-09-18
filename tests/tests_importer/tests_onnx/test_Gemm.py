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

# FIXME Where do tests go?
from tests.tests_importer import importerTestCase


class TestGemm(importerTestCase.ImporterTestCase):
    """Test for Gemm Layer."""

    def test_gemm_nn(self) -> None:
        """Test Gemm layer for config nn."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 20])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 10])

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(20, 10).astype(np.float32)
        Gemm_W_initializer = importerTestCase.create_initializer_tensor(name=Gemm_W_name,
                                                                        tensor_array=Gemm_W,
                                                                        data_type=onnx.TensorProto.FLOAT)

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(10).astype(np.float32)
        Gemm_B_initializer = importerTestCase.create_initializer_tensor(name=Gemm_B_name,
                                                                        tensor_array=Gemm_B,
                                                                        data_type=onnx.TensorProto.FLOAT)

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

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_gemm_nt(self) -> None:
        """Test Gemm layer for config nt."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 20])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 10])

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(10, 20).astype(np.float32)
        Gemm_W_initializer = importerTestCase.create_initializer_tensor(name=Gemm_W_name,
                                                                        tensor_array=Gemm_W,
                                                                        data_type=onnx.TensorProto.FLOAT)

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(10).astype(np.float32)
        Gemm_B_initializer = importerTestCase.create_initializer_tensor(name=Gemm_B_name,
                                                                        tensor_array=Gemm_B,
                                                                        data_type=onnx.TensorProto.FLOAT)

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
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_gemm_tn(self) -> None:
        """Test Gemm layer for config tn."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [20, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 10])

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(20, 10).astype(np.float32)
        Gemm_W_initializer = importerTestCase.create_initializer_tensor(name=Gemm_W_name,
                                                                        tensor_array=Gemm_W,
                                                                        data_type=onnx.TensorProto.FLOAT)

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(10).astype(np.float32)
        Gemm_B_initializer = importerTestCase.create_initializer_tensor(name=Gemm_B_name,
                                                                        tensor_array=Gemm_B,
                                                                        data_type=onnx.TensorProto.FLOAT)

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
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_gemm_tt(self) -> None:
        """Test Gemm layer for config tt."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [20, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 10])

        Gemm_W_name = "Gemm_w"
        Gemm_W = np.random.rand(10, 20).astype(np.float32)
        Gemm_W_initializer = importerTestCase.create_initializer_tensor(name=Gemm_W_name,
                                                                        tensor_array=Gemm_W,
                                                                        data_type=onnx.TensorProto.FLOAT)

        Gemm_B_name = "Gemm_B"
        Gemm_B = np.random.rand(10).astype(np.float32)
        Gemm_B_initializer = importerTestCase.create_initializer_tensor(name=Gemm_B_name,
                                                                        tensor_array=Gemm_B,
                                                                        data_type=onnx.TensorProto.FLOAT)

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
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx").layers

        self.assert_list_layers_equals(list_layers, reference)


if __name__ == "__main__":
    importerTestCase.main()
