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

import keras
import numpy as np

class TestAcasDecr128(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testAcasDecr128Keras(self):
        model = keras.models.load_model(test_path + '/models/acas/acas_decr128/acas_decr128.h5')
        testshape = (model.input.shape[1:])
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        keras_result = np.array(model.predict(dataset)).flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,test_path + '/models/acas/acas_decr128/acas_decr128.h5', self.tmpdir_name+'/dataset.txt')
        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

    def testAcasDecr128Python(self):
        model = keras.models.load_model(test_path + '/models/acas/acas_decr128/acas_decr128.h5')
        testshape = (model.input.shape[1:])
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,test_path + '/models/acas/acas_decr128/acas_decr128.h5', self.tmpdir_name+'/dataset.txt')
        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == '__main__':
    acetoneTestCase.main()