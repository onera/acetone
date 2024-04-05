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
import sys
sys.path.append("/tmp_user/ldtis203h/yaitaiss/acetone/tests")
import acetoneTestCase as acetoneTestCase

import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as rt

tf.keras.backend.set_floatx('float32')


class TestLayers(acetoneTestCase.AcetoneTestCase):
    """Test for Dense Layer"""

    
    def test_Resize_Nearest(self):
        testshape = (1,2,2)
        # IO tensors (ValueInfoProto).
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
        size_initializer = acetoneTestCase.create_initializer_tensor(name=size_name,
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
        model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model_def.opset_import[0].version = 13

        onnx.checker.check_model(model_def)
        dataset = acetoneTestCase.create_dataset(testshape)
        onnx.save(model_def, "./tmp_dir/model.onnx")
        with open('./tmp_dir/model.txt',"w+") as f:
            print(model_def, file = f)
        f.close()

        sess = rt.InferenceSession('./tmp_dir/model.onnx')
        input_name = sess.get_inputs()[0].name
        result = sess.run(None,{input_name: dataset})
        onnx_result = result[0].flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test('./tmp_dir/model.onnx', './tmp_dir/dataset.txt').flatten()
        
        self.assertListAlmostEqual(list(acetone_result), list(onnx_result))
    

    
    
if __name__ == '__main__':
    acetoneTestCase.main()