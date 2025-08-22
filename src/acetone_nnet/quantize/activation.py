import logging

import numpy as np
from typing_extensions import Self

from acetone_nnet.generator.activation_functions import ActivationFunctions, Linear
from acetone_nnet.quantize import qform
import logging

class QuantizeShiftActivation(ActivationFunctions):
    """Cast and shift quantized layer output."""

    def __init__(
        self: Self,
        ctype: str,
        pytype: str,
        qparam: str,
        qin: str,
        qout: str,
        activation_function: ActivationFunctions | None = None,
    ) -> None:
        if activation_function is None:
            activation_function = Linear()
        self.name = "qshift"
        self.comment = "and int-cast and shift output"
        self.shift = self._compute_post_shift(qparam, qin, qout)
        self.ctype = ctype
        self.pytype = pytype
        self.activation = activation_function

    def compute(self: Self, z: np.ndarray) -> np.ndarray:
        """Compute the python output."""
        out = np.right_shift(z, self.shift)
        out1 = out.astype(self.pytype)
        if (out!=out1).all():
            logging.warning(f"Q Activation shift truncated MSB {out}, {out1}")
        return self.activation.compute(out1)

    def write_activation_str(self: Self, var: str) -> str:
        """Generate the string to print."""
        return self.activation.write_activation_str(
            f"({self.ctype})({var} >> {self.shift})",
        )

    def _compute_post_shift(
        self,
        qparam: str,
        qin: str,
        qout: str,
    ) -> int:
        """Compute the rescaling factor."""
        (_, mparam) = qform.parse_q_format(qparam)
        (_, min) = qform.parse_q_format(qin)
        (_, mout) = qform.parse_q_format(qout)
        qshift = min + mparam - mout
        if qshift < 0:
            logging.warning(
                f"{self} qpost_shift invalid {qshift}, take 0",
            )
            qshift = 0
        return qshift
