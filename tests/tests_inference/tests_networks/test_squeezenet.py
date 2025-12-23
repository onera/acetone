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

import onnx
import onnxruntime as rt

from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase
import torch
import torch.nn as nn
from torch.export import export, ExportedProgram

class ReluN(nn.Module):
    def __init__(self, n=10) -> None:
        super().__init__()
        self.n = n
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x,0,self.n)

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, relun=10) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = ReluN(relun)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = ReluN(relun)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = ReluN(relun)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNetv11(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, relun=10) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            ReluN(relun),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64, relun),
            Fire(128, 16, 64, 64, relun),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128, relun),
            Fire(256, 32, 128, 128, relun),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192, relun),
            Fire(384, 48, 192, 192, relun),
            Fire(384, 64, 256, 256, relun),
            Fire(512, 64, 256, 256, relun),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            #nn.Dropout(p=dropout), 
            final_conv, ReluN(relun), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

class TestSqueezenet(acetoneTestCase.AcetoneTestCase):
    """Inference test for squeezenet model."""

    def test_aaaaaaaaaaa_squeezenet_pytorch(self) -> None:
        with torch.no_grad():
            data = torch.rand(1,3,224,224, requires_grad=False, dtype=torch.float32)
            pytorch_model = SqueezeNetv11(relun=1000)
            pytorch_model.eval()
            program : ExportedProgram = export(pytorch_model,(data,))
            acetone_result,python_result = acetoneTestCase.run_acetone_for_test(
                "test",
                program,
                data.numpy(),
                bin_dataset=True
            )
            self.assertListAlmostEqual(acetone_result, python_result)
            self.assertListAlmostEqual(pytorch_model(data).numpy()[0], python_result)

    def test_squeezenet_onnx(self) -> None:
        """Test squeezenet model, compare between keras et C code."""
        model_path = MODELS_DIR / "squeezenet1" / "squeezenet1.onnx"
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
        acetone_result,python_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            model_path,
            self.tmpdir_name + "/dataset.txt",
        )
        self.assertListAlmostEqual(acetone_result, onnx_result)
        self.assertListAlmostEqual(python_result, onnx_result)

if __name__ == "__main__":
    acetoneTestCase.main()
