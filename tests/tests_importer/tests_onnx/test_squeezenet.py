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
importerTestCase_path = '/'.join(__file__.split('/')[:-2])
import sys
sys.path.append(importerTestCase_path)
import importerTestCase

import onnx

class TestSqueeznet(importerTestCase.ImporterTestCase):
    """Test for squeezenet"""

    def test_squeezenet(self):
        model_path = '/'.join(__file__.split('/')[:-3])+'/models/squeezenet1/squeezenet1.onnx'
        model = onnx.load_model(model_path)

        reference = self.import_layers(model).layers
        list_layers = self.import_layers(model_path).layers
        
        self.assert_List_Layers_equals(list_layers, reference)

if __name__ == '__main__':
    importerTestCase.main()
