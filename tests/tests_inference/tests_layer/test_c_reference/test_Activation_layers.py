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
sys.path.append(__file__[:-70])
import acetoneTestCase as acetoneTestCase

import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, Conv2D

tf.keras.backend.set_floatx('float32')


class TestLayers(acetoneTestCase.AcetoneTestCase):
    """Test for Activations Layer"""
    
    def testReLu(self):
        testshape = (1,1,16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation='relu', bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testLeakyReLu(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='leaky_relu', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testSigmoid(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='sigmoid', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))
    
    def testTanh(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='tanh', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
        model.save(self.tmpdir_name+'/model.h5')

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,self.tmpdir_name+'/model.h5', self.tmpdir_name+'/dataset.txt')
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(acetone_result[0]), list(keras_result))

if __name__ == '__main__':
    acetoneTestCase.main()