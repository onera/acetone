"""Init file.

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


from .Reduce import Reduce
from .reduce_max.ReduceMax import ReduceMax
from .reduce_max.ReduceMaxDefault import ReduceMaxDefault
from .reduce_mean.ReduceMean import ReduceMean
from .reduce_mean.ReduceMeanDefault import ReduceMeanDefault
from .reduce_min.ReduceMin import ReduceMin
from .reduce_min.ReduceMinDefault import ReduceMinDefault
from .reduce_prod.ReduceProd import ReduceProd
from .reduce_prod.ReduceProdDefault import ReduceProdDefault
from .reduce_sum.ReduceSum import ReduceSum
from .reduce_sum.ReduceSumDefault import ReduceSumDefault

__all__ = (
    "Reduce",
    "ReduceMax",
    "ReduceMaxDefault",
    "ReduceMean",
    "ReduceMeanDefault",
    "ReduceMin",
    "ReduceMinDefault",
    "ReduceProd",
    "ReduceProdDefault",
    "ReduceSum",
    "ReduceSumDefault",
)
