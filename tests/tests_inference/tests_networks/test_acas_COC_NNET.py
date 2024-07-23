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

test_path = "/".join(__file__.split("/")[:-3])
import sys

sys.path.append(test_path+"/tests_inference")
import acetoneTestCase

from tests.tests_inference import acetoneTestCase


class TestAcas_COC_NNet(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testAcas_COC_NNet(self):
        NNet_result = [2.45578096e+04, 2.44936744e+04, 2.44764976e+04, 2.44966370e+04, 2.44581565e+04]
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, test_path + "/models/acas/acas_COC/nn_acas_COC.nnet", test_path + "/models/acas/acas_COC/test_input_acas_COC.txt")

        self.assertListAlmostEqual(list(acetone_result[0]), list(NNet_result))

    def testAcas_COC_NNet_Python(self):
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name, test_path + "/models/acas/acas_COC/nn_acas_COC.nnet")

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == "__main__":
    acetoneTestCase.main()
