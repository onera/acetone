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

import sys
import subprocess

### All the test
all = './tests/'

### testing the result of the inference
inference = all+'tests_inference/'

network = inference+'tests_networks/'

layer = inference+'tests_layer/'
c_python = layer+'test_c_python/'
c_reference = layer+'test_c_reference/'
without_template = layer+'test_without_template/'

### Testing the import from the model
importer = all+'tests_importer/'
keras = importer + 'tests_keras/'




possible_test = {'all':all, 
                    'test_importer':importer,
                        'tests_keras':keras,
                    'test_inference': inference,
                        'tests_layer':layer,
                            'test_c_reference':c_reference,
                            'test_c_python':c_python,
                        'tests_network':network}

cmd = ['python3','-m','unittest','discover']

if(len(sys.argv) == 1):
    print('Add as argument the name of the test folder you want to run, or use all for testing all of them.')
    print('The argument can be the name of a directory containing test folders (ex: "test_layer" will run "test_c_python", "test_c_reference" and "test_without_template")')
else:
    cmd += [possible_test[sys.argv[1]]]
    cmd += ['test_*.py']


    subprocess.run(cmd)