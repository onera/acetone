# Tutorials

We present here a list of notebooks detailing different fonctionalities of the framework, as well as a few application use cases.

* [ACETONE tutorial 1](./tutorial1_introduction.ipynb)
* [ACETONE tutorial 2](./tutorial2_using_variants.ipynb)
* [ACETONE tutorial 3](./tutorial3_using_debug_mode.ipynb)
* [ACAS Case Study]()

## ACETONE Tutorial 1: Introduction to the framework

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/onera/acetone/8-notebook-demo?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial1_introduction.ipynb)

After training a model, we want to embed it on a target, we therefore need an equivalent C code for the model.

In the first part of this notebook, we use the path to a **nnet** file to load the model and generate the code. We then directly use a **ONNX** model as input of our code generator before producing the required code.

In the later part, we compare the several outputs given by the framework and a chosen reference (given by **ONNX Runtime**) to ensure the preservation of semantics.



## ACETONE Tutorial 2: Using other implementations of a layer


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/onera/acetone/8-notebook-demo?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial2_using_variants.ipynb)

As the world of neural networks is constantly evolving, new computation techniques are bound to emerge in the industry. Moreover, as the final goal of the generated code is to be embedded, the computation methode must be adapted to the target. Thus, it is essential for the framework to be flexible and to offer the possibility to easily change the generated code.

This notebook offers a quick overview of ACETONE's versioning system, which allows us to freely customize each layers generated code by using the example of the convolution layer.  

In the first part of this notebook, we explore a few implementation already existing within the framework.

In the later part, we create our own implementation and add it to ACETONE.


 ## ACETONE Tutorial 3: Using the framework's debug mode

 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/onera/acetone/8-notebook-demo?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial3_using_debug_mode.ipynb)

 While trying to implement a new layer or version of an already existing layer, we are likely to encounter some difficulties, particularly when it comes to preserving semantics. However, ACETONE's generic code, as the model, only returns the global output of the model. Thus, we can't really know if the errors comes from the layer we are working on, an earlier layer, or juste a memory representation problem. Thus, the framework offers a way to have a finer look on the execution trace of the model.

 In the first part of this notebook, we use the framework's debug mode to generate the C code and gathe all the outputs.

 In the later part, we use some provided tools to create a reference (known to be true) before comparing it with the computed outputs.
 