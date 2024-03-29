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
import keras
import numpy as np
from keras.layers import Input, Conv2D, Concatenate

tf.keras.backend.set_floatx('float32')


class TestLayers(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testConcatenate(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Concatenate(axis=3)([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        dataset = acetoneTestCase.create_dataset(testshape)
        model.save('./tmp_dir/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test('./tmp_dir/model.h5', './tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)
    
if __name__ == '__main__':
    acetoneTestCase.main()