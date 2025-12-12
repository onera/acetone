from tests.common import MODELS_DIR
from tests.tests_inference import acetoneTestCase
import torch
from torch.export import export, ExportedProgram
import logging
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
        #logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')

        model_params = MODELS_DIR / "acas" / "acas_pytorch" / "classif_weights.pth"
        model = makeNN()
        model.load_state_dict(torch.load(model_params, weights_only=True))
        program:ExportedProgram = export(model,(torch.rand(5),))
        acetone_result, reference = acetoneTestCase.run_acetone_for_test(
            self.tmpdir_name,
            program,
            bin_dataset=True)
        self.assertListAlmostEqual(acetone_result, reference)