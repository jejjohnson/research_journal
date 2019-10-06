# Gaussian Processes


---
## Algorithms

### Sparse Spectrum Gaussian Processes

These are essentially the analogue to the random fourier features for Gaussian processes.

#### SSGP

1. Sparse Spectrum Gaussian Process Regression - Lázaro-Gredilla et. al. (2010) - [PDF](http://jmlr.csail.mit.edu/papers/v11/lazaro-gredilla10a.html)
   > The original algorithm for SSGP.
2. Prediction under Uncertainty in Sparse Spectrum Gaussian Processes
with Applications to Filtering and Control - Pan et. al. (2017) - [PDF](http://proceedings.mlr.press/v70/pan17a.html)
    > This is a moment matching extension to deal with the uncertainty in the inputs at prediction time.

* Python Implementation
  * [Numpy](https://github.com/marcpalaci689/SSGPR)
  * [GPFlow](https://github.com/jameshensman/VFF/blob/master/VFF/ssgp.py)


#### Variational SSGPs

So according to [this paper]() the SSGP algorithm had a tendency to overfit. So they added some additional parameters to account for the noise in the inputs making the marginal likelihood term intractable. They added variational methods to deal with the 
1. Improving the Gaussian Process Sparse Spectrum Approximation by Representing Uncertainty in Frequency Inputs - Gal et. al. (2015) 
   > "...proposed variational inference in a sparse spectrum model that is derived from a GP model." - Hensman et. al. (2018)
2. Variational Fourier Features for Gaussian Processes -  Hensman et al (2018)  
   > "...our work aims to directly approximate the posterior of the true models using a variational representation." - Hensman et. al. (2018)

* Yarin Gal's Stuff - [website](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Improving)
* Code
  * [Numpy](https://github.com/marcpalaci689/SSGPR)
  * [Theano](https://github.com/yaringal/VSSGP)

---
### Latent Variable Models

<p align="center">
  <img src="figures/lvms.png" alt="drawing" width="400"/>
</p>

**Figure**: (Gal et. al., 2015)

1. Latent Gaussian Processes for Distribution Estimation of Multivariate Categorical Data - Gal et. al. (2015) - [Resources](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Latent)

---

---

## Code

---

---




---
### Review Papers

* When GPs meet big data - [paper](https://arxiv.org/pdf/1807.01065.pdf)


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


### Notebooks (outside of docs)

* GPy
  * DeepBayes2017 - [Standard](https://github.com/bayesgroup/deepbayes2017/blob/master/sem4-GP/1_GP_basics_filled.ipynb) | [Large Scale](https://github.com/bayesgroup/deepbayes2017/blob/master/sem4-GP/3_LargeScaleGP_filled.ipynb)
  * DeepBayes2018 - [Standard](https://github.com/bayesgroup/deepbayes-2018/blob/master/day5_gp/gp_basic_filled.ipynb)
* GPyTorch
  * DeepBayes2019 - [Standard](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/GP/gp_solution.ipynb)