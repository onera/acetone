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

acetoneTestCase_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase

import numpy as np
import onnx
import onnxruntime as rt

class TestTile(acetoneTestCase.AcetoneTestCase):
    """Test for Tile Layer"""

    def testTile(self):
        in_shape =list(np.random.randint(1,50, size=3))
        in_shape = (1, int(in_shape[0]),int(in_shape[1]),int(in_shape[2]))

        repeats_name = 'repeats'
        repeats = np.random.randint(1,10, size=3)
        repeats = np.array([1,repeats[0],repeats[1],repeats[2]])
        repeats_initializer = acetoneTestCase.create_initializer_tensor(name = repeats_name,
                                                                       tensor_array = repeats,
                                                                       data_type = onnx.TensorProto.INT64)
        out_shape = (int(in_shape[i]*repeats[i]) for i in range(4))
        
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                            onnx.TensorProto.FLOAT,
                                            in_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            out_shape)
        
        Tile_node = onnx.helper.make_node(
            name = 'Tile',
            op_type = 'Tile',
            inputs = [model_input_name,repeats_name],
            outputs = [model_output_name],
        )
        graph = onnx.helper.make_graph(
            nodes = [Tile_node],
            name = 'Tile',
            inputs = [X],
            outputs = [Y],
            initializer = [repeats_initializer],
        )
        model = onnx.helper.make_model(graph)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,in_shape)
        onnx.save(model,self.tmpdir_name+'/model.onnx' )

        sess = rt.InferenceSession(self.tmpdir_name+'/model.onnx')
        input_name = sess.get_inputs()[0].name
        result = sess.run(None,{input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.onnx', self.tmpdir_name+'/dataset.txt')
        self.assertListAlmostEqual(acetone_result[0],onnx_result)

if __name__ == '__main__':
    acetoneTestCase.main()