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
from keras.layers import Input, ZeroPadding2D

from tests.tests_inference import acetoneTestCase


class TestPad(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def test_Pads(self):
        testshape = (10, 10, 3)

        input = Input(testshape)
        out = ZeroPadding2D(padding=(1, 1))(input)

        model = keras.Model(input, out)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)
        model.save(self.tmpdir_name + "/model.h5")

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            self.tmpdir_name + "/model.h5",
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )
        keras_result = np.array(model.predict(dataset)).flatten()

        self.assertListAlmostEqual(list(list(acetone_result[1])), list(keras_result))


if __name__ == "__main__":
    acetoneTestCase.main()
