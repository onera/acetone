from setuptools import setup, find_packages
from src.acetone_nnet.cli.compare import cli_compare
from src.acetone_nnet.cli.generate import cli_acetone

with open("README.md") as f:
    readme = f.read()

extensions = []

setup(
        name='acetone_nnet',
        version='0.4.dev1',
        description="Predictable programming framework for ML applications in safety-critical systems.",
        long_description=readme,
        long_description_content_type="text/markdown",
        packages=find_packages(where='./src/acetone_nnet'),
        package_data={'acetone_nnet': ["templates/**.tpl",
                                        "templates/normalization/*.tpl",
                                        "templates/memory_layout/*.tpl",
                                        "templates/layers/*.tpl",
                                        "templates/layers/Resize/*.tpl",
                                        "templates/layers/Pad/*.tpl",
                                        "templates/layers/Gemm/*.tpl",
                                        "templates/layers/Conv/*.tpl"] },
        zip_safe=False,
        ext_modules=extensions,
        python_requires="==3.10",
        cmdclass={
            "acetone_compare": cli_compare,
            "acetone_generate": cli_acetone,
        },
    )