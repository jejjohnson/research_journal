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
