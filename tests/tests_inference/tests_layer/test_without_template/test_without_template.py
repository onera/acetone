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
sys.path.append(__file__[:-75])
import acetoneTestCase as acetoneTestCase

class TestWithoutTemplate(acetoneTestCase.AcetoneTestCase):
    """Comapre the result of Acetone with the generation by template and without"""

    def testAdd(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Add/Add.h5','./tests/tests_inference/tests_layer/test_without_template/Add/test_input_Add.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Add/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testAverage(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Average/Average.h5','./tests/tests_inference/tests_layer/test_without_template/Average/test_input_Average.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Average/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testAveragePool(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/AveragePool/AveragePool.h5','./tests/tests_inference/tests_layer/test_without_template/AveragePool/test_input_AveragePool.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/AveragePool/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testConcat(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Concat/Concat.h5','./tests/tests_inference/tests_layer/test_without_template/Concat/test_input_Concat.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Concat/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testConv(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Conv/Conv.h5','./tests/tests_inference/tests_layer/test_without_template/Conv/test_input_Conv.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Conv/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testDense(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Dense/Dense.h5','./tests/tests_inference/tests_layer/test_without_template/Dense/test_input_Dense.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Dense/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testMax(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Max/Max.h5','./tests/tests_inference/tests_layer/test_without_template/Max/test_input_Max.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Max/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testMaxPool(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/MaxPool/MaxPool.h5','./tests/tests_inference/tests_layer/test_without_template/MaxPool/test_input_MaxPool.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/MaxPool/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testMin(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Min/Min.h5','./tests/tests_inference/tests_layer/test_without_template/Min/test_input_Min.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Min/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testMul(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Mul/Mul.h5','./tests/tests_inference/tests_layer/test_without_template/Mul/test_input_Mul.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Mul/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testSub(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/tests_inference/tests_layer/test_without_template/Sub/Sub.h5','./tests/tests_inference/tests_layer/test_without_template/Sub/test_input_Sub.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/Sub/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testSqueezenet(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/squeezenet1/squeezenet1.onnx','./tests/tests_inference/tests_layer/test_without_template/squeeznet/test_intput_without_template.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/squeeznet/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testLenet5(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/lenet5/lenet5_trained/lenet5_trained.h5','./tests/tests_inference/tests_layer/test_without_template/lenet5/test_intput_without_template.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/lenet5/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
    def testAcas_decr(self):

        result_with_template = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/acas/acas_decr128/acas_decr128.h5','./tests/models/acas/acas_decr128/test_input_acas_decr128.txt')
        result_without_template = acetoneTestCase.read_output_c('./tests/tests_inference/tests_layer/test_without_template/acas_decr128/output_c.txt')

        self.assertListAlmostEqual(result_with_template[0],result_without_template,atol=0)
    
if __name__ == '__main__':
    acetoneTestCase.main()