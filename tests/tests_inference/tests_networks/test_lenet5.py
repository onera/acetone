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
sys.path.append(__file__[:-46])
import acetoneTestCase as acetoneTestCase

import keras
import numpy as np

class TestLenet5(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testLenet5Keras(self):
        model = keras.models.load_model('./tests/models/lenet5/lenet5_trained/lenet5_trained.h5')
        testshape = (model.input.shape[1:])
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        keras_result = np.array(model.predict(dataset)).flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/lenet5/lenet5_trained/lenet5_trained.h5', self.tmpdir_name+'/dataset.txt')
        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

    def testLenet5Python(self):
        model = keras.models.load_model('./tests/models/lenet5/lenet5_trained/lenet5_trained.h5')
        testshape = (model.input.shape[1:])
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/lenet5/lenet5_trained/lenet5_trained.h5', self.tmpdir_name+'/dataset.txt')
        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == '__main__':
    acetoneTestCase.main()