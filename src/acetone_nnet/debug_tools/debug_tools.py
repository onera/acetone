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

def compare_result(acetone_result:list|np.ndarray, reference_result:list|np.ndarray, targets:list, verbose:bool = False):
    try:
        assert len(acetone_result) == len(reference_result)
    except AssertionError as msg:
        print("Error: both result don't have the same size")

        if verbose:
            print(msg)
        
        return False
    
    correct = True
    for i in range(len(acetone_result)):
        print('--------------------------------------------')
        print('Comparing',targets[i])
        try:
            np.testing.assert_allclose(acetone_result[i],reference_result[i],atol=1e-06)
        except AssertionError as msg:
            print("Error: output value of",targets[i],"incorrect")
            correct = False
            if verbose:
                print(msg)
        print('--------------------------------------------')

    return correct

def extract_outputs_c(path_to_output:str, data_type:str, nb_targets:int):
    list_result = []
    targets = []
    to_transpose = False
    with open(path_to_output,'r') as f:
        for i, line in enumerate(f):
    
            if i%2 == 0:
                line = line[:-1].split(' ')
                targets.append(line[0] + " " + line[1])
                to_transpose = int(line[2])
                if to_transpose:
                    shape = (int(line[3]),int(line[4]),int(line[5]))
            else:
                line = line[:-2].split(' ')
                if data_type == 'int':
                    line = list(map(int,line))
                elif data_type == 'double':
                    line = list(map(float,line)) 
                elif data_type == 'float':
                    line = list(map(np.float32,line))
                line = np.array(line)

                if to_transpose:
                    line = np.reshape(line, shape)
                    line = np.transpose(line, (1,2,0))
                    line = line.flatten()

                list_result.append(line)
    
            if i == 2*nb_targets-1:
                break
    f.close()

    return list_result, targets

def extract_outputs_python(result_python:tuple[list], targets_indice:list):
    outputs = [result_python[0][i] for i in targets_indice]
    targets = [result_python[1][i] for i in targets_indice]

    return outputs, targets