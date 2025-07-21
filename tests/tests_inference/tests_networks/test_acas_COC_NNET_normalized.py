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

import numpy as np

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase


class TestAcasCOCNNetnormalized(acetoneTestCase.AcetoneTestCase):
    """Inference test for Acas COC, NNet model with normalization."""

    @unittest.expectedFailure # Unsure about the reference values provided in the seminal paper.
    def test_acas_coc_normalized_nnet(self) -> None:
        """Tests Acas COC, NNet model with normalisation, compare between NNet et C code."""
        nnet_result = [2.46398403e+04,
                       2.44853907e+04,
                       2.44597071e+04,
                       2.44829454e+04,
                       2.44301250e+04]
        nnet_result = np.array(nnet_result)
        model_path = (
                MODELS_DIR / "acas" / "acas_COC" / "nn_acas_COC.nnet"
        )
        dataset_path = (
                MODELS_DIR / "acas" / "acas_COC" / "test_input_acas_COC.txt"
        )
        acetone_result, python_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,
                                                              model_path,
                                                              dataset_path)

        self.assertListAlmostEqual(acetone_result, nnet_result)
        self.assertListAlmostEqual(python_result, nnet_result)

if __name__ == "__main__":
    acetoneTestCase.main()
