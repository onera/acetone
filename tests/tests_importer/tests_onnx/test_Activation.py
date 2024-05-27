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


class TestConv(importerTestCase.ImporterTestCase):
    """Test for Conv Layer"""

    def testReLu(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Relu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)

    def testSigmoid(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Sigmoid",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testLeakyReLu(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="LeakyRelu",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testTanh(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Tanh",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testExp(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Exp",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testLog(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        activation_node = onnx.helper.make_node(
            op_type="Log",
            inputs=[conv1_output_name],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.onnx').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testClip(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            [ None,5, 10, 10])
        
        conv1_output_name = "output_conv1"
        conv1_in_channels = 3
        conv1_out_channels = 5
        conv1_kernel_shape = (7,7)
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
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_name],
            kernel_shape=conv1_kernel_shape,
            auto_pad='SAME_UPPER',
            strides = (1,1),
        )

        min_initializer = importerTestCase.create_initializer_tensor(name='min',
                                                                    tensor_array=np.random.rand(1),
                                                                    data_type=onnx.TensorProto.FLOAT)

        max_initializer = importerTestCase.create_initializer_tensor(name='max',
                                                                    tensor_array=np.random.rand(1)*20,
                                                                    data_type=onnx.TensorProto.FLOAT)

        activation_node = onnx.helper.make_node(
            op_type="Clip",
            inputs=[conv1_output_name,'min','max'],
            outputs=[model_output_name],
        )

        graph = onnx.helper.make_graph(
            nodes = [conv1_node,activation_node],
            name = 'Conv',
            inputs = [X],
            outputs = [Y],
            initializer = [conv1_W_initializer_tensor,conv1_B_initializer_tensor,min_initializer,max_initializer],
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