"""
 *******************************************************************************
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

acetoneTestCase_path = '/'.join(__file__.split('/')[:-2])
import sys
sys.path.append(acetoneTestCase_path)
import acetoneTestCase

class TestAcas_fully_connected_NNet_normalized(acetoneTestCase.AcetoneTestCase):
    """Test for Concatenate Layer"""

    def testAcas_fully_connectedNormalizedNNet(self):
        NNet_result = [354.385,364.375,366.455,355.707,363.078]
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/acas/acas_fully_connected/acas_fully_connected.nnet', './tests/models/acas/acas_fully_connected/test_input_acas_fully_connected.txt',normalize=True)

        self.assertListAlmostEqual(list(acetone_result[0]), list(NNet_result))
    
    def testAcas_fully_connectedNormalizedNNetPython(self):
        testshape = (5,)
        dataset = acetoneTestCase.create_dataset(self.tmpdir_name,testshape)
    
        acetone_result = acetoneTestCase.run_acetone_for_test(self.tmpdir_name,'./tests/models/acas/acas_fully_connected/acas_fully_connected.nnet', self.tmpdir_name+'/dataset.txt',normalize=True)

        self.assertListAlmostEqual(list(acetone_result[0]), list(acetone_result[1]))

if __name__ == '__main__':
    acetoneTestCase.main()