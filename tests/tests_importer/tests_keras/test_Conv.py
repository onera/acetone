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
from keras.layers import Conv2D, Input

from tests.tests_importer import importerTestCase

tf.keras.backend.set_floatx("float32")


class TestConv(importerTestCase.ImporterTestCase):
    """Test for Conv Layer."""

    def test_conv_6loops(self) -> None:
        """Test Conv2D with 6 loops implementation."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation=None,
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model,
                                       {"Conv2D": "6loops"}).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5",
                                         {"Conv2D": "6loops"}).layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_conv_indirect_gemm(self) -> None:
        """Test Conv2D with indirect_gemm implementation."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation=None,
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model,
                                       {"Conv2D": "indirect_gemm_nn"}).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5",
                                         {"Conv2D": "indirect_gemm_nn"}).layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_conv_std_gemm(self) -> None:
        """Test Conv2D with std_gemm implementation."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation=None,
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model,
                                       {"Conv2D": "std_gemm_nn"}).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5",
                                         {"Conv2D": "std_gemm_nn"}).layers

        self.assert_list_layers_equals(list_layers, reference)


if __name__ == "__main__":
    importerTestCase.main()
