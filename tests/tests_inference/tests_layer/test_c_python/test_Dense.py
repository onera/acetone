"""*******************************************************************************
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
from keras.layers import Dense, Input

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float32")


class TestDense(acetoneTestCase.AcetoneTestCase):
    """Test for Dense Layer"""

    def test_Dense1(self):
        testshape = (1, 1, 16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer="he_normal")(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

    def test_Dense2(self):
        testshape = (1, 1, 500)
        units = 250

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer="he_normal")(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              self.tmpdir_name + "/dataset.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))


if __name__ == "__main__":
    acetoneTestCase.main()
