"""Activation Functions type definition.

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

from abc import abstractmethod
from collections.abc import Callable

import numpy as np
from typing_extensions import Self


class ActivationFunctions:
    """Abstract class for activation functions."""

    def __init__(self: Self) -> None:
        """Initiate an activation function."""
        self.name = None
        self.name = ""
        self.comment = ""

    def as_lambda(self) -> Callable:
        """Return the activation function as a lambda for code generation."""
        return lambda x: self.write_activation_str(x)

    @abstractmethod
    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""

    @abstractmethod
    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""

    def __eq__(
        self: Self,
        other,
    ) -> bool:
        """Eq method for layers."""
        if type(self) is type(other):
            keys = list(self.__dict__.keys())
            for key in keys:
                if (
                    key in ("previous_layer", "next_layer")
                    or type(self.__dict__[key]) is dict
                ):
                    continue

                if type(self.__dict__[key]) is np.ndarray:
                    if (other.__dict__[key] != self.__dict__[key]).any():
                        return False
                elif other.__dict__[key] != self.__dict__[key]:
                    return False
        # compare two layers and say if they are equals
        else:
            return False
        return True


class Sigmoid(ActivationFunctions):
    """Sigmoid layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "sigmoid"
        self.comment = " and apply sigmoid function"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        isnegz = z < 0
        expmz = np.exp(z,dtype=z.dtype)
        x = expmz/(1+expmz)
        y = 1 / (1 + np.exp(-z,dtype=z.dtype))
        ''' stable algorithm: shall only compute negative input exponent '''
        return np.where(isnegz,x,y)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return f"({local_var} < 0) ? (expf({local_var}) / (1. + expf({local_var}))) : (1. / (1. + expf(-{local_var})))"

class Silu(ActivationFunctions):
    """Silu layer."""
    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "silu"
        self.comment = " and apply silu function"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return z / (1 + np.exp(-z))

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return local_var + " / (1 + exp(-" + local_var + "))"

class ReLu(ActivationFunctions):
    """ReLu layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "relu"
        self.comment = " and apply rectifier"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return np.maximum(0, z)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        # output = condition ? value_if_true : value_if_false
        return local_var + " > 0 ? " + local_var + " : 0"


class LeakyReLu(ActivationFunctions):
    """LeakyReLu layer."""

    def __init__(self: Self, alpha: float) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "leakyrelu"
        self.comment = " and apply rectifier"
        self.alpha = alpha

        ### Checking value consistency ###
        if self.alpha < 0:
            msg = "Error: alpha value in LeakyRelu (alpha < 0)"
            raise ValueError(msg)

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        temp_tensor = z.flatten()
        for i in range(len(temp_tensor)):
            if temp_tensor[i] < 0:
                temp_tensor[i] = self.alpha * temp_tensor[i]
        return temp_tensor.reshape(z.shape)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        # output = condition ? value_if_true : value_if_false
        return f"{local_var} > 0. ? {local_var} : {self.alpha!s}*{local_var}"


class TanH(ActivationFunctions):
    """TanH layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "hyperb_tan"
        self.comment = " and apply hyperbolic tangent function"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        isnegz = z < 0
        exp2z = np.exp(2.*z,dtype=z.dtype)
        expm2z = np.exp(-2.*z,dtype=z.dtype)
        x = (exp2z - 1.) / (exp2z + 1.)
        y = (1 - expm2z) / (1 + expm2z)
        ''' stable algorithm: shall only compute negative input exponent '''
        return np.where(isnegz,x,y)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return f"({local_var} < 0.) ? (expf(2.*{local_var}) - 1.) / (expf(2.*{local_var}) + 1.) : (1. - expf(-2.*{local_var})) / (1. + expf(-2.*{local_var}))"

class Linear(ActivationFunctions):
    """Linear layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "linear"
        self.comment = ""

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return z

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return local_var


class Exponential(ActivationFunctions):
    """Exponential layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "Exponential"
        self.comment = " and apply exponential function"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return np.exp(z)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return "exp(" + local_var + ")"


class Logarithm(ActivationFunctions):
    """Logarithm layer."""

    def __init__(self: Self) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "Logarithm"
        self.comment = " and apply logarithm function"

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return np.log(z)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        return "log(" + local_var + ")"


class Clip(ActivationFunctions):
    """Clip layer."""

    def __init__(self: Self, max_value: float, min_value: float) -> None:
        """Initiate the class."""
        super().__init__()
        self.name = "Clip"
        self.comment = " and apply rectifier"
        self.max = max_value
        self.min = min_value

        ### Checking value consistency ###
        if self.min > self.max:
            raise ValueError(
                "Error: min and max values in Clip ("
                + str(self.min)
                + " > "
                + str(self.max)
                + ")",
            )

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        return np.clip(z, self.min, self.max)

    def write_activation_str(self: Self, local_var: str) -> str:
        """Generate the string to print."""
        # output = condition ? value_if_true : value_if_false
        # output = input > max ? max : (input < min ? min : input)
        return (
            local_var
            + " > "
            + str(self.max)
            + " ? "
            + str(self.max)
            + " : ("
            + local_var
            + " < "
            + str(self.min)
            + " ? "
            + str(self.min)
            + " : "
            + local_var
            + ")"
        )
