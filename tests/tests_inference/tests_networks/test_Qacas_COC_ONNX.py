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
import tempfile

target_conf = """
{
  "name": "AVX",
  "cflags": "-O1",
  "quantization": {
    "dtype": "short",
    "temp_dtype": "int",
    "pydtype": "int16",
    "temp_pydtype": "int32",
    "layers": {
      "X": {
        "out": "Q0.15"
      },
      "W0": {
        "out": "Q-2.17"
      },
      "MatMul_13": {
        "out": "Q-1.16",
        "in": "Q0.15",
        "params": "Q-2.17"
      },
      "B0": {
        "out": "Q-1.16"
      },
      "Add_14": {
        "out": "Q-1.16",
        "in": "Q-1.16",
        "params": "Q-1.16"
      },
      "W1": {
        "out": "Q5.10"
      },
      "MatMul_15": {
        "out": "Q1.14",
        "in": "Q-1.16",
        "params": "Q5.10"
      },
      "B1": {
        "out": "Q1.14"
      },
      "Add_16": {
        "out": "Q1.14",
        "in": "Q1.14",
        "params": "Q1.14"
      },
      "W2": {
        "out": "Q2.13"
      },
      "MatMul_17": {
        "out": "Q2.13",
        "in": "Q1.14",
        "params": "Q2.13"
      },
      "B2": {
        "out": "Q2.13"
      },
      "Add_18": {
        "out": "Q2.13",
        "in": "Q2.13",
        "params": "Q2.13"
      },
      "W3": {
        "out": "Q2.13"
      },
      "MatMul_19": {
        "out": "Q3.12",
        "in": "Q2.13",
        "params": "Q2.13"
      },
      "B3": {
        "out": "Q3.12"
      },
      "Add_20": {
        "out": "Q3.12",
        "in": "Q3.12",
        "params": "Q3.12"
      },
      "W4": {
        "out": "Q2.13"
      },
      "MatMul_21": {
        "out": "Q3.12",
        "in": "Q3.12",
        "params": "Q2.13"
      },
      "B4": {
        "out": "Q3.12"
      },
      "Add_22": {
        "out": "Q3.12",
        "in": "Q3.12",
        "params": "Q3.12"
      },
      "W5": {
        "out": "Q14.1"
      },
      "MatMul_23": {
        "out": "Q15.0",
        "in": "Q3.12",
        "params": "Q14.1"
      },
      "B5": {
        "out": "Q15.0"
      },
      "Add_24": {
        "out": "Q15.0",
        "in": "Q15.0",
        "params": "Q15.0"
      }
    }
  }
}
"""


def writeconf(conf):
    with open("AVX512VNNI.json", "w") as f:
        f.write(conf)


class TestQAcasCOCONNX(acetoneTestCase.AcetoneTestCase):
    """Inference test for Quantized ACAS COC, ONNX model."""

    def test_Qacas_coc_onnx_python(self) -> None:
        """Tests Acas COC, ONNX model, compare between python et C code."""
        model_path = MODELS_DIR / "acas" / "acas_COC" / "nn_acas_COC.onnx"

        writeconf(target_conf)  # writes target config AVX512VNNI.json in the CWD
        fdata = np.random.rand(100,1,5).astype(np.float32) # 100 samples
        idata = np.round(fdata*(np.iinfo(np.int16).max-1)).astype(np.int16)
        c_result, py_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            target="AVX512VNNI",
            datatest_path=idata,
            bin_dataset=True
        )
        tmp = tempfile.TemporaryDirectory()
        cf_result, pyf_result = acetoneTestCase.run_acetone_for_test(
            tmp.name,
            model_path,
            datatest_path=fdata,
            bin_dataset=True
        )
        tmp.cleanup()
        self.assertListAlmostEqual(c_result, py_result)
        self.assertListAlmostEqual(cf_result, pyf_result)

        self.assertListAlmostEqual(c_result, cf_result, rtol=5.e-4)
        self.assertListAlmostEqual(py_result, pyf_result, rtol=5.e-4)

if __name__ == "__main__":
    acetoneTestCase.main()
