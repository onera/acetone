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
import torch

from tests.tests_inference import acetoneTestCase
from torch.export import export
class TestModel(torch.nn.Module):
    def __init__(self,kernel,stride=1,padding=0):
        super(TestModel, self).__init__()        
        self.conv = torch.nn.Conv2d(4,5,kernel, stride=stride, padding=padding)
        self.act = torch.nn.ReLU()
    def forward(self, x):
        return self.act(self.conv(x))

def test_HWC_conv(tester, kernel, padding=0, stride=1):
        model = TestModel(kernel,stride=stride,padding=padding)
        model = model.to(memory_format=torch.channels_last)
        data = torch.rand(1,4,10,10,requires_grad=False,dtype=torch.float32)
        data = data.to(memory_format=torch.channels_last)
        program = export(model,(data,)) # torch fx export
        with torch.no_grad():
            torch_out = model(data)
        acetone_result = acetoneTestCase.run_acetone_for_test(
            tester.tmpdir_name, 
            program,
            conv_algo="direct_block",
            bin_dataset=True,
            datatest_path=data.permute(0,2,3,1).numpy(),
            gen_data_format="channels_last"
            )
        tester.assertListAlmostEqual(acetone_result[1], torch_out.permute(0,2,3,1).numpy().ravel())
        tester.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))        


class TestConv(acetoneTestCase.AcetoneTestCase):
    def testConv_direct_bloc(self):
        test_HWC_conv(self,5)

    def testConv_direct_bloc_pad1(self):
        test_HWC_conv(self,5,1)

    def testConv_direct_bloc_pad1_str2(self):
        test_HWC_conv(self,5,1,2)

    def testConv_winograd(self):
        test_HWC_conv(self,3)

    def testConv_winograd_pad1(self):
        test_HWC_conv(self,3,1)

    def testConv_winograd_pad1_str2(self):
        test_HWC_conv(self,3,1,2)

    def testConv_1x1(self):
        test_HWC_conv(self,1)

    def testConv_1x1_str2(self):
        test_HWC_conv(self,1,stride=2)

if __name__ == "__main__":
    acetoneTestCase.main()
