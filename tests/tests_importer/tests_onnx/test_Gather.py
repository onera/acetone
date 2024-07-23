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


class TestGather(importerTestCase.ImporterTestCase):
    """Test for Gather Layer."""

    def test_gather_1(self) -> None:
        """Test gather on axis==1."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 3, 10])

        indices_name = "indices"
        indices = np.random.randint(0, 10, 3)
        indices_initializer = importerTestCase.create_initializer_tensor(
            name=indices_name,
            tensor_array=indices,
            data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=2,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.onnx").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_gather_2(self) -> None:
        """Test gather on axis==1."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 3])

        indices_name = "indices"
        indices = np.random.randint(0, 10, 3)
        indices_initializer = importerTestCase.create_initializer_tensor(
            name=indices_name,
            tensor_array=indices,
            data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indices_name],
            outputs=[model_output_name],
            axis=3,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indices_initializer],
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
