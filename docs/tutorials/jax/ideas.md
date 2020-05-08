# Jax Tutorial Ideas


## Tutorials

* `vmap`
* `jit`
* `grad`, `jacobian`, `hessian`
* `optimizers`
* containers - `Dict`, `Tuple`, `NamedTuple`

---

## Use Cases

* Gaussian Processes
* Kernel Matrices and Derivatives
* Kernel Density Estimation - [Blog](https://matthewmcateer.me/blog/gaussian-kde-from-scratch/) | 
* Optimized Kernel Ridge Regression (OKRR) - [Notebook](https://gonzmg88.github.io/Talk_OKRR/)
* Optimized Kernel Entropy Components Analysis (OKECA)
* Centered Kernel Alignment (CKA)
* Gaussianization Flows


---

## Projects

1. Uncertain Inputs for GPs
2. Gaussianization Flows - Case Study

* Uncertain Inputs for Gaussian Processes
  * Linearized GP (Taylor Expansion, 1st Order, 2nd Order)
  * Moment Matching (RBF)
    * Scratch - Loops, numba
    * Jax
  * MCMC Posterior Approximation
    * Scratch
    * Numpyro - NUTS/HMC
  * Variational GP
    * Numpyro - SVI

---
  
#### Data

* 1D Examples
* 2D Examples
* IASI Example
* Ocean Data


#### Future Stuff

* Sparse Models
* Deep Models
* DKL Models
* 

## Resources

**GPS**

* [VI 4 GPs w. Jax](https://github.com/FrancisRhysWard/Variational_inference) | [BBVI GP](https://github.com/FrancisRhysWard/Variational_inference/blob/master/distribution_prediction/blackbox_vi/blackbox_vi_gp.py)
* [JaxGP](https://github.com/salutnomo/Structured-Learning-for-Robot-Control/blob/master/GP2/JAX/jaxGP.py)
* [GPs and DGPs with Jax n Flax](https://github.com/danieljtait/ladax)

**OKECA**

* [PCA Projections](https://github.com/MinRegret/timecast/blob/master/timecast/learners/_pcr.py)