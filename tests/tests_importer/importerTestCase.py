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

import unittest
import tempfile
import numpy as np

import acetone_nnet
from acetone_nnet import Layer, ActivationFunctions

def are_layers_equals(self, other):
    #compare two layers and say if they are equals
    if type(self) != type(other):
        return False
    else:
        
        keys = list(self.__dict__.keys())
        for key in keys:
            if key == 'previous_layer':
                continue
            elif key == 'next_layer':
                continue
            elif type(self.__dict__[key]) == dict:
                continue

            if type(self.__dict__[key]) == np.ndarray:
                if (other.__dict__[key] != self.__dict__[key]).any():
                    return False
            else:
                if other.__dict__[key] != self.__dict__[key]:
                    return False
    return True

Layer.__eq__ = are_layers_equals
ActivationFunctions.__eq__ = are_layers_equals

def get_weights(layer_keras,data_type_py = np.float32):
    weights = data_type_py(layer_keras.get_weights()[0])
    if(len(weights.shape) < 3):
        for i in range(3-len(weights.shape)): 
            weights = np.expand_dims(weights, axis=-1)
    weights = np.moveaxis(weights, 2, 0)
    if(len(weights.shape) < 4):
        weights = np.expand_dims(weights, axis=0)
    return weights

class ImporterTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_name = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()
    
    def import_layers(self, file, conv_algorithm = 'std_gemm_nn'):
        return acetone_nnet.CodeGenerator(file = file,
                                          test_dataset_file = None,
                                          function_name = 'inference',
                                          nb_tests = 1,
                                          conv_algorithm = conv_algorithm,
                                          normalize = False)
    
    def assert_Layers_equals(actual, desired):

        mismatched_type = []
        if not issubclass(type(actual), acetone_nnet.Layer) :
            mismatched_type.append(actual)
        if not issubclass(type(desired), acetone_nnet.Layer) :
            mismatched_type.append(desired)
        if mismatched_type:
            err_msg = 'Class mismatch (not a subclass of Layer): ' 
            for mismatch in mismatched_type:
                err_msg += str(type(mismatch)) + ' '
            raise AssertionError(err_msg)

        if actual != desired:
            if type(actual) != type(desired):
                err_msg = 'Type mismatch: ' + type(actual).__name__ + ' and ' + type(desired).__name__
            else:
                keys = list(actual.__dict__.keys())
                mismatched_keys = []
                for attribut in keys:
                    if actual.__dict__[attribut] != desired.__dict__[attribut]:
                        mismatched_keys.append(attribut)
                err_msg = 'Attribut mismatch: \n'
                msg=[]
                for mismatch in mismatched_keys:
                    msg.append(mismatch + ' ('+str(actual.__dict__[mismatch])+' and '+str(desired.__dict__[mismatch])+')')
                err_msg += '\n'.join(msg)
            
            raise AssertionError(err_msg)
    
    def assert_List_Layers_equals(self, actual, desired, verbose=True):

        if(len(actual) != len(desired)):
            err_msg = 'Shape error: ' + str(len(actual)) + ' != ' + str(len(desired))
            raise AssertionError(err_msg)
        
        lenght = len(actual)

        mismatched_type = []
        for i in range(lenght):
            if not issubclass(type(actual[i]), acetone_nnet.Layer) :
                mismatched_type.append(actual[i])
            if not issubclass(type(desired[i]), acetone_nnet.Layer) :
                mismatched_type.append(desired[i])
        if mismatched_type:
            err_msg = 'Class mismatch (not a subclass of Layer): ' 
            for mismatch in mismatched_type:
                err_msg += str(type(mismatch)) + ' '
            raise AssertionError(err_msg)


        err_msg = []
        for i in range(lenght):
            layer1 = actual[i]
            layer2 = desired[i]
            if layer1 != layer2:
                layer_msg = 'Layer '+str(i)+'\n'
                if type(layer1) != type(layer2):
                    layer_msg += 'Type mismatch: ' + type(layer1).__name__ + ' and ' + type(layer2).__name__
                else:
                    keys = list(layer1.__dict__.keys())
                    mismatched_keys = []
                    for attribut in keys:
                        if attribut == 'previou_layer' or attribut == 'next_layer':
                           continue

                        if type(layer1.__dict__[attribut]) == np.ndarray:
                            if (layer1.__dict__[attribut] != layer2.__dict__[attribut]).any():
                                mismatched_keys.append(attribut)
                        else:
                            if layer1.__dict__[attribut] != layer2.__dict__[attribut]:
                                mismatched_keys.append(attribut)
                    layer_msg += 'Attribut mismatch: \n'
                    msg=[]
                    for mismatch in mismatched_keys:
                        msg.append(mismatch + ' ('+str(layer1.__dict__[mismatch])+' and '+str(layer2.__dict__[mismatch])+')')
                    layer_msg += '\n'.join(msg)
                err_msg.append(layer_msg)

        if(err_msg):
            if(verbose):
                nb_err = len(err_msg)
                err_msg = '\n'.join(err_msg)
                err_msg = 'Mismastch Layers: '+str(nb_err)+'/'+str(lenght)+'\n' + err_msg
                raise AssertionError(err_msg)