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

target_conf = """
{
    "name":"AVX",
    "cflags":"-O1",
    "quantization":
    {
        "dtype":"short",
        "temp_dtype":"int",
        "pydtype":"int16",
        "temp_pydtype":"int32",
        "layers":
        {
            "X":{
                "out":"Q0.15"
            },
            "W0":{
                "out":"Q-3.18"
            },
            "MatMul_13":{
                "in":"Q0.15",
                "params":"Q-3.18",
                "out":"Q2.13"
            },
            "B0": {
                "out":"Q0.15"
            },
            "Add_14":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "W1": {
                "out":"Q0.15"
            },
            "MatMul_15":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "B1": {
                "out":"Q0.15"
            },
            "Add_16":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "W2": {
                "out":"Q0.15"
            },
            "MatMul_17":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "B2": {
                "out":"Q0.15"
            },
            "Add_18":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "W3": {
                "out":"Q0.15"
            },
            "MatMul_19":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "B3": {
                "out":"Q0.15"
            },
            "Add_20":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "W4": {
                "out":"Q0.15"
            },
            "MatMul_21":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "B4": {
                "out":"Q0.15"
            },
            "Add_22":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "W5": {
                "out":"Q0.15"
            },
            "MatMul_23":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
            },
            "B5": {
                "out":"Q0.15"
            },
            "Add_24":{
                "in":"Q0.15",
                "params":"Q0.15",
                "out":"Q2.13"
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

        c_result, py_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            target="AVX512VNNI",
        )

        self.assertListAlmostEqual(c_result, py_result)


if __name__ == "__main__":
    acetoneTestCase.main()
