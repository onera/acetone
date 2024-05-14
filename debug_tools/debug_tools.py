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

import numpy as np

def compare_result(acetone_result:list, reference_result:list, targets:list, verbose:bool = False):
    try:
        assert len(acetone_result) == len(reference_result)
    except AssertionError as msg:
        print("Error: both result don't have the same size")

        if verbose:
            print(msg)
        
        return False
    
    for i in range(len(acetone_result)):
        print('--------------------------------------------')
        print('Comparing',targets[i])
        try:
            np.testing.assert_allclose(acetone_result[i],reference_result[i],atol=1e-06)
        except AssertionError as msg:
            print("Error: output value of",targets[i],"incorrect")

            if verbose:
                print(msg)
            print('--------------------------------------------')
            return False
        print('--------------------------------------------')

    return True

