from functools import reduce
from operator import mul
from typing import Any

import numpy as np
from traits.adaptation.api import register_factory
from traits.api import (
    Array,
    BaseTuple,
    DelegatesTo,
    HasTraits,
    Int,
    Interface,
    Str,
    provides,
)
from traits.trait_converters import trait_from


class Shape(BaseTuple):
    """Tensor shape traits."""

    info_text = "a tensor shape"

    def __init__(self, **metadata: Any) -> None:
        """Initialise shape of the specified type."""
        super().__init__(**metadata)
        self.base = trait_from(Int(default_value=1))

    def validate(self, obj: object, name: str, value: tuple) -> tuple:
        """Validate shape is a tuple with all values of the specified type."""
        value = super().validate(obj, name, value)
        if len(value) > 0 and all(
            self.base.validate(obj, name, v) and v > 0 for v in value
        ):
            return value
        self.error(object, name, value)
        return self.default_value


class TensorSpec(Interface):
    """Tensor specification."""

    shape = Shape()
    dtype = Str()

    @property
    def rank(self) -> int:
        """Rank of the tensor."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Size of the tensor."""
        return reduce(mul, self.shape, 1)


@provides(TensorSpec)
class Tensor(HasTraits):
    """Numpy-backed tensor."""

    #: Tensor contents
    data = Array()

    shape = DelegatesTo("data")
    dtype = DelegatesTo("data")
    size = DelegatesTo("data")

    def __init__(self, data):
        self.data = data

    @property
    def rank(self) -> int:
        """Rank of the tensor."""
        return self.data.ndim

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Convert to a numpy array."""
        # TODO copy=True and dtype ~> return a new copy
        # TODO copy=None ~> return a copy if required (e.g. by dtype)
        # TODO copy=False ~> a copy must never be made
        if copy:
            return np.copy(self.data)
        return self.data


# Provide adapter from numpy array
register_factory(Tensor, np.ndarray, TensorSpec)
