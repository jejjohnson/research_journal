# Gaussian Processes


## Code

**Numpy**

* [Sparse Spectrum Gaussian Process Regression](https://github.com/linesd/SSGPR)

**JAX**

* [Gaussian Processes in Python via Jax](https://github.com/bkompa/pygau)

**PyTorch**

* [Sparse spectrum Gaussian process regression](https://github.com/rafaol/ssgp/blob/master/ssgp/models.py)
* [Sparse Gaussian process: VFE and SVI-GP](https://github.com/Alaya-in-Matrix/SparseGP)

**Pyro**

* [PyTorch implementation of Deep Gaussian processes with various inference algorithms using probabilistic programming.](https://github.com/ahmedmalaa/Deep-Gaussian-Processes)

**TensorFlow**

* [GrandPrix GPLVM](https://github.com/ManchesterBioinference/GrandPrix)

---




---
### Review Papers


## GPU Computing

* [CuPy](https://github.com/ericmjl/bayesian-analysis-recipes/blob/master/notebooks/gp-cupy.ipynb)
* [Tensorflow]
* [PyTorch]
* [PyMC3]
* [JAX]

## Non-Stationary Kernels

* [Spatial Mapping with Gaussian Processes and NonStationary Fourier Features](https://www.sciencedirect.com/science/article/pii/S2211675317302890) - Ton et al. (2018)


## GP Extrapolation

* [Blog](https://www.danielemaasit.com/post/2018/03/19/gaussian-processes-with-spectral-mixture-kernels-to-implicitly-capture-hidden-structure-from-data/)
* [PyTorch](https://gpytorch.readthedocs.io/en/latest/examples/01_Simple_GP_Regression/Spectral_Mixture_GP_Regression.html)


---
## Sofware

* Standard - [sklearn][1] | [pymc-learn][2]
* Sparse - [pymc-learn][2] | [GPy][4] | [Pyro][3] | [GPyTorch][5]
* Advanced - [GPy][4] | [GPyTorch][5]
* GPUs
  * TensorFlow - [GPFlow][5]
  * PyTorch - [GPyTorch][5] | [Pyro][3] | [gptorch][7]
  * Theano - [pymc-learn][2]


[1]: https://scikit-learn.org/stable/modules/gaussian_process.html
[2]: https://www.pymc-learn.org/
[3]: http://pyro.ai/examples/gp.html
[4]: http://sheffieldml.github.io/GPy/
[5]: https://github.com/cornellius-gp/gpytorch
[6]: https://gpflow.readthedocs.io/en/develop/#
[7]: https://github.com/cics-nd/gptorch/


---
## Tutorials

* [An Introduction to GPR](https://juanitorduz.github.io/gaussian_process_reg/)
* http://krasserm.github.io/2018/03/19/gaussian-processes/


### Notebooks (outside of docs)

* GPy
  * DeepBayes2017 - [Standard](https://github.com/bayesgroup/deepbayes2017/blob/master/sem4-GP/1_GP_basics_filled.ipynb) | [Large Scale](https://github.com/bayesgroup/deepbayes2017/blob/master/sem4-GP/3_LargeScaleGP_filled.ipynb)
  * DeepBayes2018 - [Standard](https://github.com/bayesgroup/deepbayes-2018/blob/master/day5_gp/gp_basic_filled.ipynb)
* GPyTorch
  * DeepBayes2019 - [Standard](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/GP/gp_solution.ipynb)