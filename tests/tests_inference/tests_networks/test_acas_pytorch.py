from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase
import torch
from torch.export import export, ExportedProgram
import logging
import numpy as np
from ctypes import cdll, c_double, c_float, c_int, byref, POINTER, c_char_p

def makeNN(num_inputs=5, hidden_layers=[50]*6, num_outputs=5):

    #input
    layers = [torch.nn.Linear(num_inputs, hidden_layers[0]), torch.nn.ReLU()]

    #hidden
    for i in range(len(hidden_layers)-1):
        layers += [torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]), torch.nn.ReLU()]

    #output
    layers += [torch.nn.Linear(hidden_layers[-1], num_outputs)]

    return torch.nn.Sequential(*layers)

class TestAcasPytorch(acetoneTestCase.AcetoneTestCase):
    """Inference test for ACAS COC, pytorch model."""

    def test_acas_pytorch(self) -> None:
        logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')
        with torch.no_grad():
            model_params = MODELS_DIR / "acas" / "acas_pytorch" / "classif_weights.pth"
            model = makeNN()
            state_dic = torch.load(model_params, weights_only=True)
            model.load_state_dict(state_dic)
            data = torch.rand(5,requires_grad=False, dtype=torch.float32)
            program:ExportedProgram = export(model,(data,))
        
            acetone_result, reference = acetoneTestCase.run_acetone_for_test(
                self.tmpdir_name,
                program,
                data.unsqueeze(0).numpy(),
                bin_dataset=True)
            self.assertListAlmostEqual(acetone_result, reference)
            self.assertListAlmostEqual(model(data).numpy(), reference)

            #load the shared library for batch inference
            lib = cdll.LoadLibrary(self.tmpdir_name + '/inference.so')

            # convert input to ctypes pointer
            c_input_data = data.numpy().copy("C").ctypes.data_as(POINTER(c_float))
            # instanciate output np array and convert to ctypes
            y = np.empty(5, dtype=np.float32)
            c_output_data = y.ctypes.data_as(POINTER(c_float))
            # batch inference
            lib.batch_loop.argtypes=[POINTER(c_float), POINTER(c_float), c_int]
            lib.batch_loop(c_output_data, c_input_data, 1)
            self.assertListAlmostEqual(y, acetone_result)

            # save model parameters checkpoint
            lib.save_weights.argtypes = [c_char_p]
            lib.save_weights(bytes(self.tmpdir_name + "/checkpoint.dat",'utf-8'))

            # override shared lib model weights by state dictionary and save (weights are transposed and copy)
            lib.copy_weights(*tuple([v.numpy().T.copy("C").ctypes.data_as(POINTER(c_float)) for v in state_dic.values()]))
            lib.save_weights(bytes(self.tmpdir_name + "/checkpoint2.dat",'utf-8'))

            # readback checkpoints
            with open(self.tmpdir_name + "/checkpoint.dat","rb") as f:
                ck1 = f.read()
            with open(self.tmpdir_name + "/checkpoint2.dat","rb") as f:
                ck2 = f.read()
            # write the checkpoint reference based on model state dict
            with open(self.tmpdir_name + "/checkpoint_ref.dat","wb") as f:
                for v in state_dic.values():
                    np.ravel(v.numpy().T.copy("C")).tofile(f)
            #readback checkpoint reference
            with open(self.tmpdir_name + "/checkpoint_ref.dat","rb") as f:
                ck_ref = f.read()
            # assert save_weights is OK
            self.assertEqual(ck1,ck_ref)
            # assert copy_weights is OK
            self.assertEqual(ck2,ck_ref)
