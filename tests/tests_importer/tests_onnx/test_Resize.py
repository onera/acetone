"""
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

importerTestCase_path = '/'.join(__file__.split('/')[:-2])
import sys
sys.path.append(importerTestCase_path)
import importerTestCase

import onnx
import numpy as np


class TestResize(importerTestCase.ImporterTestCase):
    """Test for Resize Layer"""

    def testResizeNearest(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,2,2])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,7,8])

        size_name = 'size'
        size = np.array((1,1,7,8))
        size_initializer = importerTestCase.create_initializer_tensor(name=size_name,
                                                                      tensor_array=size,
                                                                      data_type=onnx.TensorProto.INT64)
        
        matmul_node_name="Resize"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="Resize",
            inputs=[model_input_name,"","",size_name],
            outputs=[model_output_name],
            mode = 'nearest',
            nearest_mode = 'round_prefer_floor',
            coordinate_transformation_mode = 'half_pixel',
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="ONNX_resize",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                size_initializer
            ],
        )

        # Create the model (ModelProto)
        model = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model.opset_import[0].version = 13

        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model,conv_algorithm='indirect_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx',conv_algorithm='indirect_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Resize_Linear(self):
        # IO tensors (ValueInfoProto).
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,4,4])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,8,8])

        size_name = 'size'
        size = np.array((1,1,8,8))
        size_initializer = importerTestCase.create_initializer_tensor(name=size_name,
                                                                      tensor_array=size,
                                                                      data_type=onnx.TensorProto.INT64)
        
        matmul_node_name="Resize"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="Resize",
            inputs=[model_input_name,"","",size_name],
            outputs=[model_output_name],
            mode = 'linear',
            nearest_mode = 'round_prefer_floor',
            coordinate_transformation_mode = 'half_pixel',
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="ONNX_resize",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                size_initializer
            ],
        )

        model = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model.opset_import[0].version = 13

        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model,conv_algorithm='indirect_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx',conv_algorithm='indirect_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Resize_Cubic(self):
        # IO tensors (ValueInfoProto).
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,4,4])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None,1,8,8])

        size_name = 'size'
        size = np.array((1,1,8,8))
        size_initializer = importerTestCase.create_initializer_tensor(name=size_name,
                                                                      tensor_array=size,
                                                                      data_type=onnx.TensorProto.INT64)
        
        matmul_node_name="Resize"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="Resize",
            inputs=[model_input_name,"","",size_name],
            outputs=[model_output_name],
            mode = 'cubic',
            nearest_mode = 'round_prefer_floor',
            coordinate_transformation_mode = 'half_pixel',
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node],
            name="ONNX_resize",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                size_initializer
            ],
        )
        
        model = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model.opset_import[0].version = 13

        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model,conv_algorithm='indirect_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx',conv_algorithm='indirect_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
if __name__ == '__main__':
    importerTestCase.main()