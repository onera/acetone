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
    def __init__(self):
        super(TestModel, self).__init__()        
        self.conv = torch.nn.Conv2d(4,5,3,padding=1)
    def forward(self, x):
        return self.conv(x)

class TestConv(acetoneTestCase.AcetoneTestCase):
    """Test for Conv Layer"""

    def testConv_direct_bloc(self):
        model = TestModel()
        model = model.to(memory_format=torch.channels_last)
        data = torch.rand(1,4,10,10,requires_grad=False,dtype=torch.float32)
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

if __name__ == "__main__":
    acetoneTestCase.main()
