"""*******************************************************************************
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

import tempfile
import unittest

import acetone_nnet
import numpy as np
import onnx
from acetone_nnet import Layer
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
) -> onnx.TensorProto:
    """Create a TensorProto."""
    return onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())


class ImporterTestCase(unittest.TestCase):
    """TestCase class for importer tests."""

    def setUp(self) -> None:
        """Create a temp_dir."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_name = self.tmpdir.name

    def tearDown(self) -> None:
        """Destroy a temp_dir."""
        self.tmpdir.cleanup()

    def import_layers(
            self,
            file: str | onnx.ModelProto | Functional | Sequential,
            versions: dict[int, str] | dict[str, str] | None = None,
    ) -> acetone_nnet.code_generator.CodeGenerator:
        """Create the CodeGenerator object from the file."""
        return acetone_nnet.CodeGenerator(file=file,
                                          test_dataset_file=None,
                                          function_name="inference",
                                          nb_tests=1,
                                          versions=versions,
                                          normalize=False)

    def assert_layers_equals(
            self,
            actual: Layer,
            desired: Layer,
    ) -> None:
        """Compare two layers."""
        mismatched_type = []
        if not issubclass(type(actual), acetone_nnet.Layer):
            mismatched_type.append(actual)
        if not issubclass(type(desired), acetone_nnet.Layer):
            mismatched_type.append(desired)
        if mismatched_type:
            err_msg = "Class mismatch (not a subclass of Layer): "
            for mismatch in mismatched_type:
                err_msg += str(type(mismatch)) + " "
            raise AssertionError(err_msg)

        if actual != desired:
            if type(actual) is type(desired):
                err_msg = ("Type mismatch: "
                           + type(actual).__name__
                           + " and "
                           + type(desired).__name__)
            else:
                keys = list(actual.__dict__.keys())
                mismatched_keys = []
                for attribute in keys:
                    if actual.__dict__[attribute] != desired.__dict__[attribute]:
                        mismatched_keys.append(attribute)
                err_msg = "Attribute mismatch: \n"
                msg = []
                for mismatch in mismatched_keys:
                    msg.append(mismatch
                               + " ("
                               + str(actual.__dict__[mismatch])
                               + " and "
                               + str(desired.__dict__[mismatch])
                               + ")")
                err_msg += "\n".join(msg)

            raise AssertionError(err_msg)

    def assert_list_layers_equals(
            self,
            actual: list,
            desired: list,
            verbose: bool = True,
    ) -> None:
        """Compare two lis of layers."""
        if len(actual) != len(desired):
            err_msg = "Shape error: " + str(len(actual)) + " != " + str(len(desired))
            raise AssertionError(err_msg)

        length = len(actual)

        mismatched_type = []
        for i in range(length):
            if not issubclass(type(actual[i]), acetone_nnet.Layer):
                mismatched_type.append(actual[i])
            if not issubclass(type(desired[i]), acetone_nnet.Layer):
                mismatched_type.append(desired[i])
        if mismatched_type:
            err_msg = "Class mismatch (not a subclass of Layer): "
            for mismatch in mismatched_type:
                err_msg += str(type(mismatch)) + " "
            raise AssertionError(err_msg)

        err_msg = []
        for i in range(length):
            layer1 = actual[i]
            layer2 = desired[i]
            if layer1 != layer2:
                layer_msg = "Layer " + str(i) + "\n"
                if type(layer1) is type(layer2):
                    layer_msg += ("Type mismatch: "
                                  + type(layer1).__name__
                                  + " and "
                                  + type(layer2).__name__)
                else:
                    keys = list(layer1.__dict__.keys())
                    mismatched_keys = []
                    for attribute in keys:
                        if attribute in ("previous_layer", "next_layer"):
                            continue

                        if type(layer1.__dict__[attribute]) is np.ndarray:
                            if (layer1.__dict__[attribute] != layer2.__dict__[attribute]).any():
                                mismatched_keys.append(attribute)
                        elif layer1.__dict__[attribute] != layer2.__dict__[attribute]:
                            mismatched_keys.append(attribute)
                    layer_msg += "Attribut mismatch: \n"
                    msg = []
                    for mismatch in mismatched_keys:
                        msg.append(mismatch
                                   + " ("
                                   + str(layer1.__dict__[mismatch])
                                   + " and "
                                   + str(layer2.__dict__[mismatch])
                                   + ")")
                    layer_msg += "\n".join(msg)
                err_msg.append(layer_msg)

        if err_msg and verbose:
            nb_err = len(err_msg)
            err_msg = "\n".join(err_msg)
            err_msg = ("Mismatch Layers: "
                       + str(nb_err)
                       + "/"
                       + str(length)
                       + "\n"
                       + err_msg)
            raise AssertionError(err_msg)
