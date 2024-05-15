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
from abc import abstractmethod

class ActivationFunctions():
    def __init__(self):
        self.name = ''
        self.comment = ''

    @abstractmethod
    def compute(self, z:np.ndarray):
        pass

    @abstractmethod
    def write_activation_str(self, local_var:str):
        pass

class Sigmoid(ActivationFunctions):
    
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'
        self.comment = ' and apply sigmoid function'
        #self.layer_type

    def compute(self, z:np.ndarray):
        return 1/(1+np.exp(-z))

    def write_activation_str(self, local_var:str):

        s = '1 / (1 + exp(-'+ local_var +'))'
        
        return s

class ReLu(ActivationFunctions):
    
    def __init__(self):
        super().__init__()
        self.name = 'relu'
        self.comment = ' and apply rectifier'
    
    def compute(self, z:np.ndarray):
        return np.maximum(0,z)

    def write_activation_str(self, local_var:str):

        s = local_var +' > 0 ? '+ local_var +' : 0' # output = condition ? value_if_true : value_if_false
        
        return s

class LeakyReLu(ActivationFunctions):
    
    def __init__(self, alpha:float):
        super().__init__()
        self.name = 'leakyrelu'
        self.comment = ' and apply rectifier'
        self.alpha = alpha

        assert self.alpha > 0
    
    def compute(self, z:np.ndarray):
        temp_tensor = z.flatten()
        for i in range(len(temp_tensor)):
            if(temp_tensor[i]<0):
                temp_tensor[i] =  self.alpha*temp_tensor[i]
        return temp_tensor.reshape(z.shape)

    def write_activation_str(self, local_var:str):

        s = local_var +' > 0 ? '+ local_var +' : '+str(self.alpha)+'*'+local_var # output = condition ? value_if_true : value_if_false
        
        return s

class TanH(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'hyperb_tan'
        self.comment = ' and apply hyperbolic tangent function'
        
    
    def compute(self, z:np.ndarray):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    def write_activation_str(self, local_var:str):

        s = '(exp('+ local_var +')-exp(-'+ local_var +'))/(exp('+ local_var +')+exp(-'+ local_var +'))'
        
        return s

class Linear(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'linear'
        self.comment = ''
    
    def compute(self, z:np.ndarray):
        return z

    def write_activation_str(self, local_var:str):

        s = local_var
        
        return s
    
class Exponential(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'Exponential'
        self.comment = ' and apply exponential function'
    
    def compute(self, z:np.ndarray):
        return np.exp(z)
    
    def write_activation_str(self, local_var:str):

        s = 'exp('+local_var+')'

        return s
    
class Logarithm(ActivationFunctions):
    def __init__(self):
        super().__init__()
        self.name = 'Logarithm'
        self.comment = ' and apply logarithm function'
    
    def compute(self, z:np.ndarray):
        return np.log(z)
    
    def write_activation_str(self, local_var:str):
        
        s = 'log('+local_var+')'
        
        return s

class Clip(ActivationFunctions):
    def __init__(self, max:float|int, min:float|int):
        super().__init__()
        self.name = 'Clip'
        self.comment = ' and apply rectifier'
        self.max = max
        self.min = min
    
    def compute(self, z:np.ndarray):
        return np.clip(z,self.min,self.max)
    
    def write_activation_str(self,local_var):
        s = local_var +' > '+str(self.max)+' ? '+ str(self.max) +' : (' + local_var + ' < ' + str(self.min) + ' ? ' + str(self.min) + ' : ' + local_var + ')'
        return s