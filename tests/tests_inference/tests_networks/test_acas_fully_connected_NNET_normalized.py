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
import numpy as np

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase


class TestAcasFullyConnectedNNetNormalized(acetoneTestCase.AcetoneTestCase):
    """Inference test for Acas fully connected, NNet model with normalization."""

    def test_acas_fully_connected_normalized_nnet(self) -> None:
        """Tests Acas fully connected, NNet model with normalisation, compare between NNet et C code."""
        nnet_result = [354.385, 364.375, 366.455, 355.707, 363.078]
        nnet_result = np.array(nnet_result)

        model_path = (
                MODELS_DIR / "acas" / "acas_fully_connected" / "acas_fully_connected.nnet"
        )
        dataset_path = (
                MODELS_DIR / "acas" / "acas_fully_connected" / "test_input_acas_fully_connected.txt"
        )
        acetone_result, python_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            dataset_path,
            normalize=True,
        )

        self.assertListAlmostEqual(acetone_result, nnet_result)
        self.assertListAlmostEqual(python_result, nnet_result)

if __name__ == "__main__":
    acetoneTestCase.main()
