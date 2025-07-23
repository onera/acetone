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

import onnx
import onnxruntime as rt

import acetone_nnet
from acetone_nnet import debug
from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase


class TestAcasCOCONNX(acetoneTestCase.AcetoneTestCase):
    """Inference test for ACAS COC, ONNX model."""

    def test_acas_coc_onnx(self) -> None:
        """Tests Acas COC, ONNX model, compare between onnx et C code."""
        model_path = MODELS_DIR / "acas" / "acas_COC" / "nn_acas_COC.onnx"
        model = onnx.load(model_path)
        testshape = tuple(
            model.graph.input[0].type.tensor_type.shape.dim[i].dim_value
            for i in range(len(model.graph.input[0].type.tensor_type.shape.dim))
        )
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, testshape)

        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        onnx_result = result[0].ravel().flatten()
        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            self.tmpdir_name + "/dataset.txt",
        )
        self.assertListAlmostEqual(acetone_result[0], onnx_result)

    def test_acas_coc_onnx_python_ref(self) -> None:
        """Tests Acas COC, ONNX model, compare between onnx et C code."""
        model_path = MODELS_DIR / "acas" / "acas_COC" / "nn_acas_COC.onnx"
        onnx_reference = onnx.load(str(model_path))
        onnx_model = onnx.load(str(model_path))

        dataset_shape = tuple(
            onnx_model.graph.input[0].type.tensor_type.shape.dim[i].dim_value
            for i in range(len(onnx_model.graph.input[0].type.tensor_type.shape.dim))
        )
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name, dataset_shape)

        # Compute ONNX debug outputs
        onnx_model, onnx_layers, onnx_outputs = debug.debug_onnx(
            target_model=onnx_model,
            dataset=dataset[0],
        )

        # Compute Acetone debug outputs
        generator = acetone_nnet.CodeGenerator(file=model_path,
                                               test_dataset=dataset,
                                               nb_tests=1,
                                               debug_mode="onnx")
        acetone_outputs, acetone_layers = generator.compute_inference("")
        acetone_outputs, acetone_layers = debug.reorder_outputs(acetone_outputs, acetone_layers)

        # Compute Acetone reference output
        _, acetone_reference = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            self.tmpdir_name + "/dataset.txt",
            run_generated=False,
        )

        # Compute ONNX reference output
        sess = rt.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dataset[0]})
        reference_output = result[0].ravel().flatten()

        self.assertListAlmostEqual(onnx_outputs[-1], reference_output)
        self.assertListAlmostEqual(acetone_outputs[-1], reference_output)
        self.assertListAlmostEqual(acetone_outputs[-1], acetone_reference)
        self.assertListAlmostEqual(acetone_reference, reference_output)

        # Compare
        same = debug.compare_result(
            acetone_result=acetone_outputs,
            reference_result=onnx_outputs,
            targets=acetone_layers,
            verbose=True,
        )

        assert same

    def test_acas_coc_onnx_python(self) -> None:
        """Tests Acas COC, ONNX model, compare between python et C code."""
        model_path = MODELS_DIR / "acas" / "acas_COC" / "nn_acas_COC.onnx"

        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
        )

        self.assertListAlmostEqual(acetone_result[0], acetone_result[1])


if __name__ == "__main__":
    acetoneTestCase.main()
