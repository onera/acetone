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
from abc import ABC, abstractmethod

class ActivationFunctions():
    def __init__(self):
        self.name = ''

    @abstractmethod
    def compute():
        pass

    @abstractmethod
    def generate_activation_c_files():
        pass

    @abstractmethod
    def write_activation_str(local_var):
        pass

class Sigmoid(ActivationFunctions):
    
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'
        #self.layer_type     
    def compute(self, z):
        return 1/(1+np.exp(-z))

    def generate_activation_c_files(self, data_type, activation_source_file, activation_header_file):

        activation_source_file.write(data_type + " sigmoid ("+data_type+" x)\n{\n    return 1 / (1 + exp(-x));\n}\n\n")
        activation_header_file.write(data_type +" sigmoid("+data_type+" x);\n")

        return

    def write_activation_str(self, local_var):

        s = '1 / (1 + exp(-'+ local_var +'))'
        
        return s

class ReLu(ActivationFunctions):
    
    def __init__(self):
        super().__init__()
        self.name = 'relu'
    
    def compute(self, z):
        return np.maximum(0,z)

    def generate_activation_c_files(self, data_type, activation_source_file, activation_header_file):
        
        activation_source_file.write(data_type + " relu ("+data_type+" x)\n{\n    if (x < 0)\n        return 0;\n    else\n        return x;\n}\n\n")
        activation_header_file.write(data_type +" relu("+data_type+" x);\n")
        
        return

    def write_activation_str(self, local_var):

        s = local_var +' > 0 ? '+ local_var +' : 0' # output = condition ? value_if_true : value_if_false
        
        return s

class TanH(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'hyperb_tan'
        
    
    def compute(self, z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    def generate_activation_c_files(self, data_type, activation_source_file, activation_header_file):

        activation_source_file.write(data_type + " hyperb_tan ("+data_type+" x)\n{\n    return (exp(x)-exp(-x))/(exp(x)+exp(-x));\n}\n\n")
        activation_header_file.write(data_type +" hyperb_tan("+data_type+" x);\n")

        return

    def write_activation_str(self, local_var):

        s = '(exp('+ local_var +')-exp(-'+ local_var +'))/(exp('+ local_var +')+exp(-'+ local_var +'))'
        
        return s

class Linear(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'linear'
    
    def compute(self, z):
        return z

    def generate_activation_c_files(self, data_type, activation_source_file, activation_header_file):

        activation_source_file.write(data_type + " linear ("+data_type+" x)\n{\n    return x;\n}\n\n")
        activation_header_file.write(data_type +" linear("+data_type+" x);\n")

        return

    def write_activation_str(self, local_var):

        s = local_var
        
        return s