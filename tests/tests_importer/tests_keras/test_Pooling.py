"""Test suite for Pooling layer on Keras importer.

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

import keras
import tensorflow as tf
from keras.layers import AveragePooling2D, Input, MaxPooling2D

from tests.tests_importer import importerTestCase

tf.keras.backend.set_floatx("float32")


class TestPooling(importerTestCase.ImporterTestCase):
    """Test for Pooling Layer."""

    def test_max_pooling(self):
        testshape = (10, 10, 3)
        pool_size = (3, 3)
        strides = (1, 1)

        inp = Input(testshape)
        out = MaxPooling2D(pool_size=pool_size, strides=strides, padding="valid", data_format="channels_last")(inp)

        model = keras.Model(inp, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_List_Layers_equals(list_layers, reference)

    def test_average_pooling2D(self):
        testshape = (10, 10, 3)
        pool_size = (3, 3)
        strides = (1, 1)

        inp = Input(testshape)
        out = AveragePooling2D(pool_size=pool_size, strides=strides, padding="valid", data_format="channels_last")(
            inp)

        model = keras.Model(inp, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_List_Layers_equals(list_layers, reference)
