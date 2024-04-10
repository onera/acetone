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
import tempfile

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, MaxPooling2D, AveragePooling2D

tf.keras.backend.set_floatx('float32')


class TestPooling(acetoneTestCase.AcetoneTestCase):
    """Test for Pooling Layer"""

    def testMaxPooling(self):
        testshape = (10,10,3)
        pool_size = (3, 3)
        strides = (1,1)

        input = Input(testshape)
        out = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))
   
    def testAveragePooling2D(self):
        testshape = (10,10,3)
        pool_size = (3, 3)
        strides = (1,1)

        input = Input(testshape)
        out = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid',data_format='channels_last')(input)

        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))    

if __name__ == '__main__':
    acetoneTestCase.main()