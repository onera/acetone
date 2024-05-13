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

acetoneTestCase_path = '/'.join(__file__.split('/')[:-2])
import sys
sys.path.append(acetoneTestCase_path)
import importerTestCase

import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Multiply,Maximum,Minimum,Average,Add,Subtract

tf.keras.backend.set_floatx('float32')


class TestBroadcast(importerTestCase.ImporterTestCase):
    """Test for Broadcast Layer"""
    
    def test_Add(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Add()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Multiply(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Multiply()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Subtract(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Subtract()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Average(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Average()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Maximum(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Maximum()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)
    
    def test_Minimum(self):
        testshape = (10,10,3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer='he_normal', padding='same',data_format='channels_last')(input)
        out = Minimum()([x1,x2])
        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name+'/model.h5').layers
        
        self.assert_List_Layers_equals(list_layers, reference)


if __name__ == '__main__':
    importerTestCase.main()