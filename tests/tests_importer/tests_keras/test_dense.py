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
from keras.layers import Input, Dense

tf.keras.backend.set_floatx('float32')


class TestDense(importerTestCase.ImporterTestCase):
    """Test for Dense Layer"""

    def test_Dense1(self):
        testshape = (1,1,16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer='he_normal')(input)

        model = keras.Model(input,out)
        model.save(self.tmpdir_name+'/model.h5')

        reference = self.import_layers(model).layers

        code_gen = self.import_layers(self.tmpdir_name+'/model.h5')
        list_layers = code_gen.layers

        print(list_layers[1] == reference[1])


if __name__ == '__main__':
    importerTestCase.main()