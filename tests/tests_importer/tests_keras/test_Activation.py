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

importerTestCase_path = '/'.join(__file__.split('/')[:-2])
import sys
sys.path.append(importerTestCase_path)
import importerTestCase

import tensorflow as tf
import keras
from keras.layers import Input, Conv2D

tf.keras.backend.set_floatx('float32')


class TestActivation(importerTestCase.ImporterTestCase):
    """Test for Activation Layers"""

    def testReLu(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model,'std_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5','std_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testLeakyReLu(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='leaky_relu', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model,'std_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5','std_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testSigmoid(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='sigmoid', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model,'std_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5','std_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def testTanh(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation='tanh', bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model,'std_gemm_nn').layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5','std_gemm_nn').layers
        
        self.assert_List_Layers_equals(list_layers, reference)


if __name__ == '__main__':
    importerTestCase.main()