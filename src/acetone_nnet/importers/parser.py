"""General parser ofr ACETONE.

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

from pathlib import Path
from sys import version_info
from typing import Any

import onnx

if (3, 12) > version_info >= (3, 10):
    from keras.engine.functional import Functional
    from keras.engine.sequential import Sequential

from acetone_nnet.ir import Layer

from torch.export import ExportedProgram

def parser(
        file_to_parse: Any,
        *,
        normalize: bool = False,
) -> (list[Layer], str, type, str, int, dict[int, int]):
    """Load a model and return the corresponding ACETONE representation."""
    if isinstance(file_to_parse, str | Path):
        # Retrieve extension for file path
        if isinstance(file_to_parse, str):
            extension = file_to_parse
        elif isinstance(file_to_parse, Path):
            extension = file_to_parse.suffix
        else:
            extension = ""

        if "json" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .JSON_importer.parser_JSON import load_json
            return load_json(str(file_to_parse))

        if "onnx" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .ONNX_importer.parser_ONNX import load_onnx
            return load_onnx(file_to_parse)

        if (3, 12) > version_info >= (3, 10) and "h5" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .H5_importer.parser_h5 import load_keras
            return load_keras(str(file_to_parse))

        if "nnet" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .NNET_importer.parser_NNET import load_nnet
            return load_nnet(str(file_to_parse), normalize)

        m = f"Error: model description .{extension[-4:]} not supported\n"
        m += "Only description supported are: .nnet,"
        if (3, 12) > version_info >= (3, 10):
            m += " .h5,"
        m+= " .json, .onnx"
        raise TypeError(m)

    # TODO Conditional import/test if onnx is available
    if type(file_to_parse) is onnx.ModelProto:
        from .ONNX_importer.parser_ONNX import load_onnx
        return load_onnx(file_to_parse)
    
    if type(file_to_parse) is ExportedProgram:
        from .PYTORCH_importer.parse_pytorch import load_pytorch
        return load_pytorch(file_to_parse)
    
    # TODO Conditional import/test if keras is available
    if ((3, 12) > version_info >= (3, 10) and
            (type(file_to_parse) is Functional or type(file_to_parse) is Sequential)):
        from .H5_importer.parser_h5 import load_keras
        return load_keras(file_to_parse)

    m = f"Error: model description {type(file_to_parse)} not supported\n"
    m += "Only description supported are: .nnet,"
    if (3, 12) > version_info >= (3, 10):
        m += " .h5,"
    m += " .json, .onnx"
    raise TypeError(m)
