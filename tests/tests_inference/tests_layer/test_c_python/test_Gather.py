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


class TestGather(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testGather1(self):
        testshape = (1, 3, 10, 10)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 1, 10, 10])

        indice_name = "indice"
        indice = np.random.randint(0, 3, (1))
        indice_initializer = acetoneTestCase.create_initializer_tensor(name=indice_name,
                                                                       tensor_array=indice,
                                                                       data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indice_name],
            outputs=[model_output_name],
            axis=1,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indice_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGather2(self):
        testshape = (1, 3, 10, 10)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 3, 10])

        indice_name = "indice"
        indice = np.random.randint(0, 10, (3))
        indice_initializer = acetoneTestCase.create_initializer_tensor(name=indice_name,
                                                                       tensor_array=indice,
                                                                       data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indice_name],
            outputs=[model_output_name],
            axis=2,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indice_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGather3(self):
        testshape = (1, 3, 10, 10)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, 3, 10, 3])

        indice_name = "indice"
        indice = np.random.randint(0, 10, (3))
        indice_initializer = acetoneTestCase.create_initializer_tensor(name=indice_name,
                                                                       tensor_array=indice,
                                                                       data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indice_name],
            outputs=[model_output_name],
            axis=-1,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indice_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def testGather2D(self):
        testshape = (10, 10)

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [10, 10])
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [3, 10])

        indice_name = "indice"
        indice = np.random.randint(0, 10, (3))
        indice_initializer = acetoneTestCase.create_initializer_tensor(name=indice_name,
                                                                       tensor_array=indice,
                                                                       data_type=onnx.TensorProto.INT32)

        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            inputs=[model_input_name, indice_name],
            outputs=[model_output_name],
            axis=0,
        )
        graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name="Gather",
            inputs=[X],
            outputs=[Y],
            initializer=[indice_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        onnx.save(model, self.tmpdir_name + "/model.onnx")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.onnx",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
if __name__ == "__main__":
    acetoneTestCase.main()
