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
from format_importer.JSON_importer.parser_JSON import load_json
from format_importer.ONNX_importer.parser_ONNX import load_onnx
from format_importer.NNET_importer.parser_NNET import load_nnet
from format_importer.H5_importer.JSON_from_keras_model import JSON_from_keras_model

def get_path(file, new_type):
    new_path = ""
    list_path = file.split("/")
    file_name = list_path[-1].split(".")[0]

    for dir in list_path[:-1]:
        new_path += dir+"/"
    
    new_path += file_name + "." + new_type
    return new_path

def parser(file_to_parse, conv_algorithm, normalize=False):

    if("json" in  file_to_parse[-4:]):
        return load_json(file_to_parse, conv_algorithm)
    
    elif("onnx" in file_to_parse[-4:]):
        return load_onnx(file_to_parse, conv_algorithm)
    
    elif("h5" in file_to_parse[-4:]):
        model = keras.models.load_model(file_to_parse)

        print("Creating the .json file...")
        new_path = get_path(file_to_parse,"json")
        JSON_from_keras_model(model, new_path)
        print("File created\n")

        return load_json(new_path, conv_algorithm)
    
    elif("nnet" in file_to_parse[-4:]):
        return load_nnet(file_to_parse,normalize)
    
    else:
        print("\nError: model description ."+file_to_parse[-4:]+" not supported")
        raise TypeError("Error: model description ."+file_to_parse[-4:]+" not supported\nOnly description supported are: .nnet, .h5, .json, .onnx\n")