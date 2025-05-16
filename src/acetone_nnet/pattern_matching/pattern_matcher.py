"""Main pattern matching function.

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

from acetone_nnet.generator.Layer import Layer
from acetone_nnet.pattern_matching.Pattern import Pattern, list_patterns


def pattern_matcher(model: list[Layer]) -> tuple[list[Layer], str]:
    """Apply pattern matching to the given model."""
    temp = model.copy()
    log = ""
    i = 0
    while i < len(temp):
        layer = temp[i]
        msg = None
        for pattern in list_patterns:
            if pattern.is_pattern(layer):
                pattern.apply_pattern(index=i,layers=temp)
                log += msg

    return temp, log

