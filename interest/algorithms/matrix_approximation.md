

* [Why are Big Data Matrices Approximately Low Rank](https://epubs.siam.org/doi/10.1137/18M1183480)


## Large Scale Kernels

* Kernel Ridge Regression
    * Standard
    * Nystrom Method
    * Random Kitchen Sinks (RKS)
    * Random Fourier Features (RFF)
    * FastFood

### Random Features

**Different Kernels**

* RBF
* Laplace
* Cauchy
* Matern12
* Matern23
* Matern52
* ArcCosine
* FastFood (RBF, Spectral Mixture)


---
## Randomized Algorithms


* Randomized Matrix Decompositions using R - Erichson (2016) - [Paper]()
* RSVDPACK: Subroutines for computing partial singular value decompositions via randomized sampling on single core, multi-core, and GPU architectures - Voronin & Martinsson (2015) - [Paper]()
* Finding structure with randomness: Probabilistic algorithms for construction approximation matrix decompositions - Halko et al (2011)


### Libraries

* [istretto](https://github.com/erichson/ristretto)
  > Includes the staple algorithms: rSVD, CUR, LU, NMF and Interpolative. Doesn't use any GPU tech.

