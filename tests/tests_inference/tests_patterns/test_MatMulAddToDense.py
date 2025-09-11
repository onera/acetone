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
import tensorflow as tf

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestMatMulAddToDense(acetoneTestCase.AcetoneTestCase):
    """Test for MatMulAddToDense pattern."""

    def testMatMulWeightRightAddToDense(self):
        testshape =int(np.random.randint(5,20))
        len_graph = int(np.random.randint(1,20))
        sizes = [testshape]
        sizes.extend(list(map(int,np.random.randint(10,100,size=len_graph))))



        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [testshape])
        model_output_name = f"add_node_{len_graph-1}"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [sizes[-1]])

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i],sizes[i+1]).astype(np.float32)
            add_B = np.random.rand(sizes[i+1]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            add_B_name = f"add_node_{i}_W"
            add_B_initializer = acetoneTestCase.create_initializer_tensor(
                name=add_B_name,
                tensor_array=add_B,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.extend([matmul_W_initializer, add_B_initializer])

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[input_name, matmul_W_name],
                outputs=[matmul_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[matmul_name, add_B_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])


    def testMatMulWeightLeftAddToDense(self):
        testshape =int(np.random.randint(5,20))
        len_graph = int(np.random.randint(1,20))
        sizes = [testshape]
        sizes.extend(list(map(int,np.random.randint(10,100,size=len_graph))))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [testshape])
        model_output_name = f"add_node_{len_graph-1}"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [sizes[-1]])

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i+1],sizes[i]).astype(np.float32)
            add_B = np.random.rand(sizes[i+1]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            add_B_name = f"add_node_{i}_W"
            add_B_initializer = acetoneTestCase.create_initializer_tensor(
                name=add_B_name,
                tensor_array=add_B,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.extend([matmul_W_initializer, add_B_initializer])

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[matmul_W_name,input_name],
                outputs=[matmul_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[matmul_name, add_B_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])


    def testMatMultAddToDenseMatMul2D(self):
        testshape = (1, 3, 10, 10)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [None, 3, 10, 10]
        )
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [None, 3, 10, 50]
        )

        matmul_W = np.random.rand(10,50).astype(np.float32)
        matmul_W_name = "W0"
        matmul_W_initializer_tensor = acetoneTestCase.create_initializer_tensor(
            matmul_W_name, matmul_W, onnx.TensorProto.FLOAT
        )

        matmul_node_name = "Matmul"
        matmul_output_name = "output_matmul"
        matmul_node = onnx.helper.make_node(
            name=matmul_node_name,
            op_type="MatMul",
            inputs=[model_input_name, matmul_W_name],
            outputs=[matmul_output_name],
        )

        add_B = np.random.rand(3,10,50).astype(np.float32)
        add_B_name = "add_node_W"
        add_B_initializer = acetoneTestCase.create_initializer_tensor(
            name=add_B_name,
            tensor_array=add_B,
            data_type=onnx.TensorProto.FLOAT,
        )

        add_node = onnx.helper.make_node(
            op_type="Add",
            inputs=[matmul_output_name, add_B_name],
            outputs=[model_output_name],
        )


        graph = onnx.helper.make_graph(
            nodes=[matmul_node, add_node],
            name="ONNX_matmul",
            inputs=[X],
            outputs=[Y],
            initializer=[
                matmul_W_initializer_tensor,
                add_B_initializer,
            ],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])

    def testMatMulAddToDenseReluAfterMatMul(self):
        testshape = int(np.random.randint(5, 20))
        len_graph = int(np.random.randint(1, 20))
        sizes = [testshape]
        sizes.extend(list(map(int, np.random.randint(10, 100, size=len_graph))))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [testshape]
        )
        model_output_name = f"add_node_{len_graph - 1}"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [sizes[-1]]
        )

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i + 1], sizes[i]).astype(np.float32)
            add_B = np.random.rand(sizes[i + 1]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            add_B_name = f"add_node_{i}_W"
            add_B_initializer = acetoneTestCase.create_initializer_tensor(
                name=add_B_name,
                tensor_array=add_B,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.extend([matmul_W_initializer, add_B_initializer])

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[matmul_W_name, input_name],
                outputs=[matmul_name],
            )
            activation_node_name = f"output_relu_{i}"
            activation_node = onnx.helper.make_node(
                op_type="Relu",
                inputs=[matmul_name],
                outputs=[activation_node_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[activation_node_name, add_B_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node,activation_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])

    def testMatMulAddToDenseTwoParentsAdd(self):
        testshape = int(np.random.randint(5, 20))
        len_graph = int(np.random.randint(1, 20))
        sizes = [testshape]
        sizes.extend(list(map(int, np.random.randint(10, 100, size=len_graph))))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [testshape]
        )
        model_output_name = f"add_node_{len_graph - 1}"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [sizes[-1]]
        )

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i + 1], sizes[i]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.append(matmul_W_initializer)

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[matmul_W_name, input_name],
                outputs=[matmul_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[matmul_name, matmul_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])


    def testMatMulAddToDenseLayerBetweenMAtMulAdd(self):
        testshape = int(np.random.randint(5, 20))
        len_graph = int(np.random.randint(1, 20))
        sizes = [testshape]
        sizes.extend(list(map(int, np.random.randint(10, 100, size=len_graph))))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [testshape]
        )
        model_output_name = f"add_node_{len_graph - 1}"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [sizes[-1]]
        )

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            softmax_name = f"softmax_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i + 1], sizes[i]).astype(np.float32)
            add_B = np.random.rand(sizes[i + 1]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            add_B_name = f"add_node_{i}_W"
            add_B_initializer = acetoneTestCase.create_initializer_tensor(
                name=add_B_name,
                tensor_array=add_B,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.extend([matmul_W_initializer, add_B_initializer])

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[matmul_W_name, input_name],
                outputs=[matmul_name],
            )
            softmax_node = onnx.helper.make_node(
                op_type="Softmax",
                inputs=[matmul_name],
                outputs=[softmax_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[softmax_name, add_B_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node, softmax_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])

    def testMatMulAddToDenseLayerAddCstNull(self):
        testshape = int(np.random.randint(5, 20))
        len_graph = int(np.random.randint(1, 20))
        sizes = [testshape]
        sizes.extend(list(map(int, np.random.randint(10, 100, size=len_graph))))

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(
            model_input_name, onnx.TensorProto.FLOAT, [testshape]
        )
        model_output_name = f"add_node_{len_graph - 1}"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [sizes[-1]]
        )

        initializers = []
        nodes = []
        input_name = model_input_name
        for i in range(len_graph):
            matmul_name = f"matmul_node_{i}"
            add_name = f"add_node_{i}"

            matmul_W = np.random.rand(sizes[i + 1], sizes[i]).astype(np.float32)
            add_B = np.zeros(sizes[i + 1]).astype(np.float32)
            matmul_W_name = f"matmul_node_{i}_W"
            matmul_W_initializer = acetoneTestCase.create_initializer_tensor(
                name=matmul_W_name,
                tensor_array=matmul_W,
                data_type=onnx.TensorProto.FLOAT,
            )
            add_B_name = f"add_node_{i}_W"
            add_B_initializer = acetoneTestCase.create_initializer_tensor(
                name=add_B_name,
                tensor_array=add_B,
                data_type=onnx.TensorProto.FLOAT,
            )
            initializers.extend([matmul_W_initializer, add_B_initializer])

            matmul_node = onnx.helper.make_node(
                op_type="MatMul",
                inputs=[matmul_W_name, input_name],
                outputs=[matmul_name],
            )
            add_node = onnx.helper.make_node(
                op_type="Add",
                inputs=[matmul_name, add_B_name],
                outputs=[add_name],
            )
            nodes.extend([matmul_node, add_node])
            input_name = add_name

        graph = onnx.helper.make_graph(
            nodes=nodes,
            name="MatmulAddToDense",
            inputs=[X],
            outputs=[Y],
            initializer=initializers,
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, [testshape])
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result_norm = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/classic",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=False,
        )
        acetone_result_opti = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name + "/optimized",
            self.tmpdir_name + "/model.onnx",
            self.tmpdir_name + "/dataset.txt",
            optimization=True,
        )

        self.assertListAlmostEqual(acetone_result_norm[1], acetone_result_opti[1])
        self.assertListAlmostEqual(acetone_result_norm[0], acetone_result_opti[0])


if __name__ == "__main__":
    acetoneTestCase.main()
