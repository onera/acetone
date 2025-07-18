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
import unittest

import keras
import tensorflow as tf
from keras.layers import (
    Add,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Input,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    Subtract,
    ZeroPadding2D,
)

from tests.tests_inference import acetoneTestCase

tf.keras.backend.set_floatx("float64")

class TestType(acetoneTestCase.AcetoneTestCase):
    """Test for Activations Layer"""

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testReLu(self):
        testshape = (1, 1, 16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation="relu", bias_initializer="he_normal")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testLeakyReLu(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation="leaky_relu", bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testSigmoid(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation="sigmoid", bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testTanh(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation="tanh", bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def test_Dense1(self):
        testshape = (1, 1, 16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer="he_normal")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def test_Dense2(self):
        testshape = (1, 1, 500)
        units = 250

        input = Input(testshape)
        out = Dense(units, activation=None, bias_initializer="he_normal")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testBatchNorm(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = BatchNormalization(axis=-1, gamma_initializer="he_normal", beta_initializer="he_normal",
                                 moving_mean_initializer="he_normal", moving_variance_initializer="ones")(x1)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testAdd(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Add()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testMul(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Multiply()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testSub(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Subtract()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testAvg(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Average()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testMax(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Maximum()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testMin(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Minimum()([x1, x2])
        model = keras.Model(inputs=[input], outputs=out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testConcatenate(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        x2 = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                    padding="same", data_format="channels_last")(input)
        out = Concatenate(axis=3)([x1, x2])

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testConv_6loops(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              conv_algo="6loops")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testConv_indirect_gemm_nn(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5",
                                                              conv_algo="indirect_gemm_nn")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testConv_std_gemm_nn(self):
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters, kernel_size=kernel_size, activation=None, bias_initializer="he_normal",
                     padding="same", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def test_Pads(self):
        testshape = (10, 10, 3)

        input = Input(testshape)
        out = ZeroPadding2D(padding=(1, 1))(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testMaxPooling(self):
        testshape = (10, 10, 3)
        pool_size = (3, 3)
        strides = (1, 1)

        input = Input(testshape)
        out = MaxPooling2D(pool_size=pool_size, strides=strides, padding="valid", data_format="channels_last")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def testAveragePooling2D(self):
        testshape = (10, 10, 3)
        pool_size = (3, 3)
        strides = (1, 1)

        input = Input(testshape)
        out = AveragePooling2D(pool_size=pool_size, strides=strides, padding="valid", data_format="channels_last")(
            input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)

    @unittest.SkipTest # Tolerance for float64 testing might be too low
    def test_Softmax(self):
        testshape = (1, 1, 16)
        units = 8

        input = Input(testshape)
        out = Dense(units, activation="softmax", bias_initializer="he_normal")(input)

        model = keras.Model(input, out)

        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, self.tmpdir_name + "/model.h5")

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1], atol=1e-15)


if __name__ == "__main__":
    acetoneTestCase.main()
