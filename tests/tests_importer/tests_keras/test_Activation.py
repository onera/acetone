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


class TestActivation(importerTestCase.ImporterTestCase):
    """Test for Activation Layers."""

    def test_relu(self) -> None:
        """Test for Relu."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation="relu",
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_leaky_relu(self) -> None:
        """Test for Leaky Relu."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation="leaky_relu",
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_sigmoid(self) -> None:
        """Test for Sigmoid."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation="sigmoid",
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_list_layers_equals(list_layers, reference)

    def test_tanh(self) -> None:
        """Test for TanH."""
        testshape = (10, 10, 3)
        filters = 3
        kernel_size = (3, 3)

        input = Input(testshape)
        out = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation="tanh",
                     bias_initializer="he_normal",
                     padding="same",
                     data_format="channels_last",
                     )(input)

        model = keras.Model(input, out)
        model.save(self.tmpdir_name + "/model.h5")

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(self.tmpdir_name + "/model.h5").layers

        self.assert_list_layers_equals(list_layers, reference)


if __name__ == "__main__":
    importerTestCase.main()
