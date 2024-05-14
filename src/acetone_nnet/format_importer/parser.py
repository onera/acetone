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

import keras
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
import onnx

from .JSON_importer.parser_JSON import load_json
from .ONNX_importer.parser_ONNX import load_onnx
from .NNET_importer.parser_NNET import load_nnet
from .H5_importer.parser_h5 import load_keras

def parser(file_to_parse:str|onnx.ModelProto|Functional|Sequential, conv_algorithm:str, normalize:bool=False, debug:None|str=None):

    if(type(file_to_parse) == str):
        if("json" in  file_to_parse[-4:]):
            return load_json(file_to_parse, conv_algorithm)
        
        elif("onnx" in file_to_parse[-4:]):
            return load_onnx(file_to_parse, conv_algorithm, debug)
        
        elif("h5" in file_to_parse[-4:]):
            return load_keras(file_to_parse, conv_algorithm, debug)
        
        elif("nnet" in file_to_parse[-4:]):
            return load_nnet(file_to_parse,normalize)
        
        else:
            print("\nError: model description ."+file_to_parse[-4:]+" not supported")
            raise TypeError("Error: model description ."+file_to_parse[-4:]+" not supported\nOnly description supported are: .nnet, .h5, .json, .onnx\n")
    
    elif(type(file_to_parse) == onnx.ModelProto):
        return load_onnx(file_to_parse, conv_algorithm, debug)
    
    elif(type(file_to_parse) == Functional or type(file_to_parse) == Sequential):
        return load_keras(file_to_parse, conv_algorithm, debug)

    else:
        print("\nError: model description .",type(file_to_parse),"not supported")
        raise TypeError("Error: model description .",type(file_to_parse),"not supported\nOnly description supported are: .nnet, .h5, .json, .onnx\n")