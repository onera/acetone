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


class TestPooling(importerTestCase.ImporterTestCase):
    """Test for Resize Layer"""

    def testAveragePooling(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])

        conv1_node = onnx.helper.make_node(
            op_type="AveragePool",
            inputs=[model_input_name],
            outputs=[model_output_name],
            auto_pad='SAME_LOWER',
            strides = (1,1),
            kernel_shape = (3,3),
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node],
            name = 'Pooling',
            inputs = [X],
            outputs = [Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testMaxPooling(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])

        conv1_node = onnx.helper.make_node(
            op_type="MaxPool",
            inputs=[model_input_name],
            outputs=[model_output_name],
            pads=[1,1,1,1],
            strides = (1,1),
            kernel_shape = (3,3),
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node],
            name = 'Pooling',
            inputs = [X],
            outputs = [Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testGlobalPooling(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,64, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,64, 1, 1])

        conv1_node = onnx.helper.make_node(
            op_type="GlobalAveragePool",
            inputs=[model_input_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node],
            name = 'Pooling',
            inputs = [X],
            outputs = [Y],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)

if __name__ == '__main__':
    importerTestCase.main()