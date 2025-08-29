""""Exporter to h5 file.

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
import onnx

from acetone_nnet.ir import Layer

from .ONNX_exporter.exporter_onnx import onnx_exporter


def exporter(
        format: str,
        list_layer: list[Layer],
        datatype_py:np.dtype,
        graph_name: str = "ACETONE_graph",
) -> onnx.ModelProto | None:
    """Export ACETONE internal representation to an model in the specified format."""
    if format=="onnx":
        return onnx_exporter(list_layer, datatype_py, graph_name)

    m = f"Unsupported format {format}"
    raise ValueError(m)