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
import unittest


class TestConv_HWC(acetoneTestCase.AcetoneTestCase):
    def testConv_HWC(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()   
                #test with K geater than 64 for 3x3 and 512 for 1x1     
                self.conv = torch.nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3)
                self.conv2 = torch.nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
                self.conv3 = torch.nn.Conv2d(64,128,kernel_size=1, stride=2)
                self.conv4 = torch.nn.Conv2d(128,512,kernel_size=3, stride=1, padding=1)
                self.act = torch.nn.ReLU()
            def forward(self, x):
                return self.conv4(self.conv3(self.act(self.conv2(self.act(self.conv(x))))))
        model = TestModel()
        model = model.to(memory_format=torch.channels_last)
        data = torch.rand(1,3,28,28,requires_grad=False,dtype=torch.float32)
        data = data.to(memory_format=torch.channels_last)
        program = export(model,(data,)) # torch fx export
        with torch.no_grad():
            torch_out = model(data)
        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name, 
            program,
            conv_algo="direct_block",
            bin_dataset=True,
            datatest_path=data.permute(0,2,3,1).numpy(),
            gen_data_format="channels_last"
            )
        self.assertListAlmostEqual(acetone_result[1], torch_out.permute(0,2,3,1).numpy().ravel())
        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))        

    def testConv_depth_wise(self):
        class TestDepthWise(torch.nn.Module):
            def __init__(self, dim):
                super(TestDepthWise, self).__init__()
                self.conv = torch.nn.Conv2d(dim,dim,kernel_size=1, stride=1, padding=0, groups=dim)
            def forward(self, x):
                return self.conv(x)
        dim=4
        model = TestDepthWise(dim)
        model = model.to(memory_format=torch.channels_last)
        data = torch.rand(1,dim,8,8,requires_grad=False,dtype=torch.float32)
        data = data.to(memory_format=torch.channels_last)
        program = export(model,(data,)) # torch fx export
        with torch.no_grad():
            torch_out = model(data)
        acetone_result = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name, 
            program,
            conv_algo="direct_block",
            bin_dataset=True,
            datatest_path=data.permute(0,2,3,1).numpy(),
            gen_data_format="channels_last"
            )
        self.assertListAlmostEqual(list(acetone_result[0]), list(torch_out.permute(0,2,3,1).numpy().ravel()))        
#        self.assertListAlmostEqual(acetone_result[1], torch_out.permute(0,2,3,1).numpy().ravel())

if __name__ == "__main__":
    acetoneTestCase.main()
