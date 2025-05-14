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
from typing import Any

import onnx
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
from acetone_nnet.generator import Layer


def parser(
        file_to_parse: Any,
        debug: None | str = None,
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
            return load_onnx(file_to_parse, debug)

        if "h5" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .H5_importer.parser_h5 import load_keras
            return load_keras(str(file_to_parse), debug)

        if "nnet" in extension[-4:]:
            # FIXME Path-based functions should take a path or string
            from .NNET_importer.parser_NNET import load_nnet
            return load_nnet(str(file_to_parse), normalize)

        print(f"\nError: model description . {extension[-4:]} not supported")
        raise TypeError("Error: model description ." + extension[
                                                       -4:] + " not supported\nOnly description supported are: .nnet, .h5, .json, .onnx\n")

    # TODO Conditional import/test if onnx is available
    if type(file_to_parse) is onnx.ModelProto:
        from .ONNX_importer.parser_ONNX import load_onnx
        return load_onnx(file_to_parse, debug)

    # TODO Conditional import/test if keras is available
    if type(file_to_parse) is Functional or type(file_to_parse) is Sequential:
        from .H5_importer.parser_h5 import load_keras
        return load_keras(file_to_parse, debug)

    print("\nError: model description .", type(file_to_parse), "not supported")
    raise TypeError("Error: model description .", type(file_to_parse),
                    "not supported\nOnly description supported are: .nnet, .h5, .json, .onnx\n")
