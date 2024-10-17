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
import numpy as np

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase


class TestLenet5(acetoneTestCase.AcetoneTestCase):
    """Inference test for lenet5 model."""

    def test_lenet5_keras(self) -> None:
        """Test lenet5 model, compare between keras et C code."""
        model_path = MODELS_DIR / "lenet5" / "lenet5_trained" / "lenet5_trained.h5"
        model = keras.models.load_model(model_path)
        testshape = (model.input.shape[1:])
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        keras_result = np.array(model.predict(dataset)).flatten()
        acetone_result, python_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            self.tmpdir_name + "/dataset.txt",
        )
        self.assertListAlmostEqual(acetone_result, keras_result)
        self.assertListAlmostEqual(python_result, keras_result)


if __name__ == "__main__":
    acetoneTestCase.main()
