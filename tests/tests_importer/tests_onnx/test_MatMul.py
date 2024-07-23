"""Test suite for Matmul layer on ONNX layer.

*******************************************************************************
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

from tests.tests_importer import importerTestCase


class TestMatMul(importerTestCase.ImporterTestCase):
    """Test for MatMul Layer."""

    def test_matmul1(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 1, 1, 5])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 1, 1, 50])

        matmul_W = np.random.rand(5, 50).astype(np.float32)
        matmul_W_name = "W0"
        matmul_W_initializer_tensor = importerTestCase.create_initializer_tensor(matmul_W_name,
                                                                                 matmul_W,
                                                                                 onnx.TensorProto.FLOAT)

        matmul_node_name = "Matmul"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="MatMul",
            inputs=[model_input_name, matmul_W_name],
            outputs=[model_output_name],
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="ONNX_matmul",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                matmul_W_initializer_tensor,
            ],
        )

        # Create the model (ModelProto)
        model = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model.opset_import[0].version = 13

        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model, conv_algorithm="indirect_gemm_nn").layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx", conv_algorithm="indirect_gemm_nn").layers

        self.assert_List_Layers_equals(list_layers, reference)

    def test_matmul2(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 1, 5, 1])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 1, 50, 1])

        matmul_W = np.random.rand(50, 5).astype(np.float32)
        matmul_W_name = "W0"
        matmul_W_initializer_tensor = importerTestCase.create_initializer_tensor(matmul_W_name,
                                                                                 matmul_W,
                                                                                 onnx.TensorProto.FLOAT)

        matmul_node_name = "Matmul"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="MatMul",
            inputs=[matmul_W_name, model_input_name],
            outputs=[model_output_name],
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="ONNX_matmul",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                matmul_W_initializer_tensor,
            ],
        )

        # Create the model (ModelProto)
        model = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model.opset_import[0].version = 13

        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model, conv_algorithm="indirect_gemm_nn").layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx", conv_algorithm="indirect_gemm_nn").layers

        self.assert_List_Layers_equals(list_layers, reference)
