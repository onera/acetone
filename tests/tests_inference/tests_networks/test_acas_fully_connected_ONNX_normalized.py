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

test_path = '/'.join(__file__.split('/')[:-3])
import sys
sys.path.append(test_path + "/tests_inference")
import acetoneTestCase

import onnx
import onnxruntime as rt

class TestAcas_fully_connected_ONNX_normalized(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testAcas_fully_connected_NormalizedONNX(self):
        model = onnx.load(test_path +'/models/acas/acas_fully_connected/acas_fully_connected_normalized.onnx')
        testshape = tuple(model.graph.input[0].type.tensor_type.shape.dim[i].dim_value for i in range(0,len(model.graph.input[0].type.tensor_type.shape.dim)))
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        sess = rt.InferenceSession(test_path +'/models/acas/acas_fully_connected/acas_fully_connected_normalized.onnx')
        input_name = sess.get_inputs()[0].name
        result = sess.run(None,{input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,test_path +'/models/acas/acas_fully_connected/acas_fully_connected_normalized.onnx', self.tmpdir_name+'/dataset.txt')

        self.assertListAlmostEqual(list(acetone_result[0]), list(onnx_result))
    
    def testAcas_fully_connected_NormalizedONNXPython(self):
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,test_path +'/models/acas/acas_fully_connected/acas_fully_connected_normalized.onnx')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == '__main__':
    acetoneTestCase.main()