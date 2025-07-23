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

from tests.tests_inference import acetoneTestCase

from onnxscript import opset18 as op
from onnxscript import script
import onnx
import logging
from onnxscript import FLOAT

targetconf = '''
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
            "Input_layer_0":{
                "out":"Q0.15"
            },
            "Input_layer_1":{
                "out":"Q0.15"
            },
            "Add_2":{
                "in":"Q0.15",
                "params":"Q0.10",
                "out":"Q0.13"
            }
        }
    }  
}
'''

def writeconf(conf):
    ''' writes target config file to CWD '''
    with open('AVX512VNNI.json','w') as f:
        f.write(conf)

@script(default_opset=op)
def ONNX_TestAdd(A: FLOAT[8],B: FLOAT[8]) -> FLOAT[8]:
    ''' ONNX script to define Add model '''
    return A + B

class TestBroadcast(acetoneTestCase.AcetoneTestCase):

    def testQAdd(self):
        """Test for QAdd Layer"""
        writeconf(targetconf)
        model = ONNX_TestAdd.to_model_proto()
        model = onnx.shape_inference.infer_shapes(model)
        logging.info(ONNX_TestAdd.to_model_proto())
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, ONNX_TestAdd.to_model_proto(),target="AVX512VNNI")
        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == "__main__":
    acetoneTestCase.main()
