# Kernel Methods

- [Software](#software)
- [Kernel Ridge Regression](#kernel-ridge-regression)


---
## Software

* Multiple Kernel Learning - [MKLpy](https://github.com/IvanoLauriola/MKLpy)
* Kernel Methods - [kernelmethods](https://github.com/raamana/kernelmethods)
* [pykernels](https://github.com/gmum/pykernels/tree/master)
    > A huge suite of different python kernels.
* [kernpy](https://github.com/oxmlcs/kerpy)
  > Library focused on statistical tests
* [keops](http://www.kernel-operations.io/keops/index.html)
  > Use kernel methods on the GPU with autograd and without memory overflows. Backend of numpy and pytorch.
* [pyGPs]()
  > This is a GP library but I saw quite a few graph kernels implemented with different Laplacian matrices implemented.
* [megaman]()
  > A library for large scale manifold learning. I saw quite a few different Laplacian matrices implemented.

---
## Kernel Ridge Regression

### Randomized Fourier Features


* [Lecture I](https://vimeo.com/237274729) 
* [Lecture II](https://video.ias.edu/csdm/2018/0212-ChristopherMusco) (Non-Linear Dimensionality Reduction)
* Data Dependent RFF - [Paper](https://arxiv.org/pdf/1711.09783.pdf)


#### Code

* [Revrand](https://github.com/NICTA/revrand/blob/master/revrand/basis_functions.py) | [Aboleth]()
  > Suite of basis functions. Very modular and easy to understand. Good suite of different randomized kernel functions. Random kernels [notebook](https://github.com/NICTA/revrand/blob/master/demos/random_kernels.ipynb) for Revrand and [Regression Master Class](https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html) with Aboleth.
* sd

---
## Dimension Reduction

### Kernel Entropy Components

* [Implementation](https://github.com/tsterbak/kernel_eca/blob/master/kernel_eca/kernel_eca.py)


---
## Support Vector Machines


### Using PyTorch

We can use 



* Deep Learning using Linear Support Vector Machines - Tang (2013) - [paper](https://arxiv.org/abs/1306.0239)
* SVM with PyTorch - [Blog](http://bytepawn.com/svm-with-pytorch.html)
  > A good blog with a nice and simple walk-through.
* [Github Code](https://github.com/kazuto1011/svm-pytorch)
  > This includes the hinge loss and the l2 penalty in the training procedure.
