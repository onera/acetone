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

import test.acetoneTestCase as acetoneTestCase

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate

tf.keras.backend.set_floatx('float32')


class TestLayers(acetoneTestCase.AcetoneTestCase):
    """Test for Dense Layer"""

    
    def test_Dense1(self):
        testshape = (1,1,16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(testshape)


        acetone_result = acetoneTestCase.run_acetone_for_test(model,'./tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result), list(keras_result))
    

    def test_Dense2(self):
        testshape = (1,1,500)
        units = 250

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(testshape)


        acetone_result = acetoneTestCase.run_acetone_for_test(model,'./tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)
    
    def testConv(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        model = keras.Model(input,out)

        dataset = acetoneTestCase.create_dataset(testshape)

        acetone_result = acetoneTestCase.run_acetone_for_test(model, './tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)
    
    def testMaxPooling(self):
        testshape = (10,10,3)
        pool_size = (3, 3)
        strides = (1,1)

        input = Input(testshape)
        out = MaxPooling2D(pool_size=pool_size, strides=strides, padding='valid',data_format='channels_last')(input)
        model = keras.Model(input,out)

        dataset = acetoneTestCase.create_dataset(testshape)

        acetone_result = acetoneTestCase.run_acetone_for_test(model, './tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)
    
    
    def testAveragePooling2D(self):
        testshape = (10,10,3)
        pool_size = (3, 3)
        strides = (1,1)

        input = Input(testshape)
        out = AveragePooling2D(pool_size=pool_size, strides=strides, padding='valid',data_format='channels_last')(input)
        model = keras.Model(input,out)

        dataset = acetoneTestCase.create_dataset(testshape)

        acetone_result = acetoneTestCase.run_acetone_for_test(model, './tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)

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

        acetone_result = acetoneTestCase.run_acetone_for_test(model, './tmp_dir/dataset.txt').flatten()
        keras_result = np.array(model.predict(dataset)).flatten()
        self.assertListAlmostEqual(acetone_result,keras_result)
    
if __name__ == '__main__':
    acetoneTestCase.main()