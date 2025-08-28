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

import acetone_nnet
import onnx
import onnxruntime as rt

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase
import unittest

class TestYolo(acetoneTestCase.AcetoneTestCase):
    """Inference test for squeezenet model."""

    @unittest.expectedFailure
    def test_yolo_onnx(self) -> None:
        """Test squeezenet model, compare between keras et C code."""
        model_path = MODELS_DIR / "yolo" / "tinyyolov2-7.onnx"
        model = onnx.load(model_path)
        testshape = tuple(
            model.graph.input[0].type.tensor_type.shape.dim[i].dim_value
            for i in range(1, len(model.graph.input[0].type.tensor_type.shape.dim))
        )
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset})
        onnx_result = result[0].ravel().flatten()
        acetone_result, python_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            self.tmpdir_name + "/dataset.txt",
        )
        self.assertListAlmostEqual(acetone_result, onnx_result, rtol = 5e-04, atol = 5e-04,)
        self.assertListAlmostEqual(python_result, onnx_result, rtol = 5e-04, atol = 5e-04)


if __name__ == "__main__":
    acetoneTestCase.main()
