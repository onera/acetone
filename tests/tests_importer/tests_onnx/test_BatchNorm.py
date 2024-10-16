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


class TestBatchNorm(importerTestCase.ImporterTestCase):
    """Test for BatchNorm Layer."""

    def test_batch_norm(self) -> None:
        """Test BatchNorm layer."""
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 5, 10, 10])

        conv1_output_node_name = "Conv1_Y"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7, 7)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                 *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT)
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT)

        conv1_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name,
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad="SAME_UPPER",
            strides=(1, 1),
        )

        scale_name = "scale"
        scale = np.random.rand(5)
        scale_initializer = importerTestCase.create_initializer_tensor(name=scale_name,
                                                                       tensor_array=scale,
                                                                       data_type=onnx.TensorProto.FLOAT)

        bias_name = "bias"
        bias = np.random.rand(5)
        bias_initializer = importerTestCase.create_initializer_tensor(name=bias_name,
                                                                      tensor_array=bias,
                                                                      data_type=onnx.TensorProto.FLOAT)

        mean_name = "mean"
        mean = np.random.rand(5)
        mean_initializer = importerTestCase.create_initializer_tensor(name=mean_name,
                                                                      tensor_array=mean,
                                                                      data_type=onnx.TensorProto.FLOAT)

        var_name = "var"
        var = np.random.rand(5)
        var_initializer = importerTestCase.create_initializer_tensor(name=var_name,
                                                                     tensor_array=var,
                                                                     data_type=onnx.TensorProto.FLOAT)

        gather_node = onnx.helper.make_node(
            name="BatchNormalization",
            op_type="BatchNormalization",
            inputs=[conv1_output_node_name, scale_name, bias_name, mean_name, var_name],
            outputs=[model_output_name],
            epsilon=1e-05,
        )

        graph = onnx.helper.make_graph(
            nodes=[conv1_node, gather_node],
            name="BatchNormalization",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor,
                         conv1_B_initializer_tensor,
                         scale_initializer,
                         bias_initializer,
                         mean_initializer,
                         var_initializer],
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
