import sys
import subprocess

### testing the result of the inference
network = 'tests/tests_inference/tests_networks/'
c_python = 'tests/tests_inference/tests_layer/test_c_python/'
c_reference = 'tests/tests_inference/tests_layer/test_c_reference/'
without_template = 'tests/tests_inference/tests_layer/test_without_template/'

layer = 'tests/tests_inference/tests_layer/'
inference = 'tests/tests_inference/'

### Testing the import from the model
importer = ''


### All the test
all = './'


possible_test = {'all':all, 
                    'test_importer':importer, 
                    'test_inference': inference,
                        'test_layer':layer,
                            'test_without_template':without_template,
                            'test_c_reference':c_reference,
                            'test_c_python':c_python,
                        'test_network':network}

cmd = ['python3','-m','unittest','discover']

if(len(sys.argv) == 1):
    print('Add as argument the name of the test folder you want to run, or use all for testing all of them.')
    print('The argument can be the name of a directory containing test folders (ex: "layer" will run "test_c_python", "test_c_reference" and "test_without_template")')
else:
    cmd += [possible_test[sys.argv[1]]]
    cmd += ['test_*.py']
    print(cmd)


    subprocess.run(cmd)