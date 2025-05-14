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

from .average_pooling.AveragePooling2D import AveragePooling2D
from .average_pooling.AveragePooling2DDefault import AveragePooling2DDefault
from .max_pooling.MaxPooling2D import MaxPooling2D
from .max_pooling.MaxPooling2DDefault import MaxPooling2DDefault
from .Pooling2D import Pooling2D

__all__ = (
    "AveragePooling2D",
    "AveragePooling2DDefault",
    "MaxPooling2D",
    "MaxPooling2DDefault",
    "Pooling2D",
)
