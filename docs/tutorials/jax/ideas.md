# Jax Tutorial Ideas


## Main Ideas

1. Introduction
   > Linear Regression, Regularization, 
   > vmap, jit,
   > grad, jacobian, hessian
2. Gaussian Processes
   > Exact GP, Kernel, MLE, 

#### Comparing Methods

* Exact GP
* Linearized GP (Taylor Expansion, 1st Order, 2nd Order)
* Moment-Matching GP
  * Scratch - Loops, numba
  * Jax - vmap, jit
* MCMC GP
  * Scratch - Collin
  * numpyro - NUTS/HMC
* Variational GP
  * scratch
  * Idea - Pyro
  * numpyro - SVI
  
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





---

## 1. Using Jax

Basics:
* grad
* jit
* vmap
* jacobian
* hessian
* params
* optimizers


## 2. Regression Master Class


## 3. Kernel Methods

* Kernel Least Squares
* Gaussian Processes
  * [Krasserm](http://krasserm.github.io/2018/03/19/gaussian-processes/)
* Support Vector Machines


## 4. Special Algorithms

* Optimized Kernel Ridge Regression
  * [Gonzalo's Notebook](https://gonzmg88.github.io/Talk_OKRR/)
* Optimized Kernel Entropy Components Analysis

## 5. Input Uncertainty

* Taylor Expansion
* Moment Matching
* Variational
* MCMC (NUTS/HMC)
* Heteroscedastic Likelihood