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


class TestBroadcast(importerTestCase.ImporterTestCase):
    """Test for Broadcast Layer"""

    def testAdd(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Add",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Add",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testSub(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        #create the Merging Node: Sub
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Sub",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Sub",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testMul(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        #create the Merging Node: Mul
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Mul",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Mul",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testDiv(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        #create the Merging Node: Div
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Div",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Div",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testMax(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        #create the Merging Node: Max
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Max",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Max",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testMin(self):
        model_input_name = "X"
        model_input_channels = 1
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_input_channels, 32, 32])
        model_output_name = "Y"
        model_output_channels =60
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [None, model_output_channels, 32,32])
        
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv1_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )

        conv2_output_node_name = 'Conv2_Y'
        conv1_in_channels = model_input_channels
        conv1_out_channels = model_output_channels
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv2_W = np.random.rand(conv1_out_channels, conv1_in_channels,
                                *conv1_kernel_shape).astype(np.float32)
        conv2_B = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = importerTestCase.create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                model_input_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[conv2_output_node_name],
            kernel_shape=conv1_kernel_shape,
            pads=conv1_pads,
        )
        
        #create the Merging Node: Min
        merging_node = onnx.helper.make_node(
            name="Merging_node",
            op_type="Min",
            inputs=[conv1_output_node_name,conv2_output_node_name],
            outputs=[model_output_name]
        )
        

        # Create the graph (GraphProto)
        graph = onnx.helper.make_graph(
            nodes=[conv1_node,conv2_node,merging_node],
            name="Min",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
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