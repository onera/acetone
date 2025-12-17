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

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase
import numpy as np
import unittest
class TestLSTM_ONNX(acetoneTestCase.AcetoneTestCase):
    """Inference test for Quantized ACAS COC, ONNX model."""

    @unittest.skip("need a model to test")
    def test_lstm_onnx_python(self) -> None:
        model_path = MODELS_DIR / "lstm" / "lstm.onnx"
        fdata = np.random.rand(16,1,22).astype(np.float32)
        c_result, py_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            datatest_path=fdata
        )
        self.assertListAlmostEqual(c_result, py_result)