"""Test suite for Decr128 model on Keras importer.

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

from tests.common import MODELS_DIR
from tests.tests_importer import importerTestCase

tf.keras.backend.set_floatx("float32")


class TestDecr128(importerTestCase.ImporterTestCase):
    """Test for Decr128 model."""

    def test_decr128(self):
        model_path = (
                MODELS_DIR / "acas" / "acas_decr128" / "acas_decr128.h5"
        )
        model = keras.models.load_model(model_path)

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(model_path).layers

        self.assert_List_Layers_equals(list_layers, reference)
