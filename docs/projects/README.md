# Project Webpages

* Author: J. Emmanuel Johnson

---

**Table Of Contents**

- [Similarity Measures](#similarity-measures)
  - [Main Projects](#main-projects)
  - [Software](#software)
  - [Applications](#applications)
- [Uncertainty Quantification](#uncertainty-quantification)
  - [Main Projects](#main-projects-1)
  - [Software](#software-1)
  - [Applications](#applications-1)
- [Derivatives of Kernel Methods](#derivatives-of-kernel-methods)
  - [Main Project](#main-project)
  - [Applications](#applications-2)
- [Side Projects](#side-projects)

---

## [Similarity Measures](/projects/similarity/README.md)

### Main Projects


**[HSIC and Kernel Parameters](projects/similarity/rbf_params/README.md)**


In this study, I investigate the kernel parameters and the different methods we can use to estimate the kernels of these dependence measures. I assess the validity of each one by doing a full empirical study on 1D distributions and on N-D distributions. 

* [Main Project](projects/similarity/rbf_params/README.md)
* [HSIC Implementations](projects/similarity/README.md)
* [Visualization](projects/similarity/README.md)

---

### [Software]()


**PySim**

A python package that I use as my base in order to explore different aspects of similarity measures. Some highlights include entropy, mutual information and KL-Divergence. I explore a few methods including K-Nearest Neighbors, kernel density estimation, Hilbert-Schmidt Criterion and Gaussianization methods.

---

### [Applications]()

**Drought Factors**


**Information Plane in Neural Networks**

---

## Uncertainty Quantification


### Main Projects


**[Input Uncertainty](projects/egps/README.md)**

* Assuming Input Error in the methods
* Extensive Literature review
* Looking at linearized and variational methods to propagate the errors through the GPs
* Applications to IASI satellite data and prediction
* Explore some relations to Kalman Filtering


**Gaussianization**


* [Main Project](projects/rbig/README.md)
  
---

### Software

**RBIG 1.1**

This is a python package which a revamped API that makes it easier to prototype and try out different methods of Gaussianization. It is based off of the [deep-density-destructors]() repository but much simpler and scaled down for my needs. It's open source to allow other researchers to take a look at how one can do Gaussianization.

* [Github Repo]()


**RBIG 2.0**

This is the next iteration for Gaussianization to allow one to train it on GPUs. In this, I explore different ways we can use the normalizing flows architecture to do Gaussianization transformations.

* [Github Repo]()


---

### Applications


**[Spatial-Temporal Representations](projects/rbig/README.md)**

**CMIP5 Climate Model Comparisons**


---

## Derivatives of Kernel Methods

### Main Project


* [Main Project Page]()
* [Github Repository]()
* [Jax Tutorial for Kernel Methods]()
* [Application to GP input Uncertainty]()

---

### Applications

**Sensitivity Analysis for Emulation**

---

## Side Projects

**Machine Learning for Multi-Output Ocean Applications**

* Multi-Output Gaussian Processes
* High-Dimensional, Multi-Output Data

---




