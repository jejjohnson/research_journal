# Literature Review

- [Theory](#theory)
  - [Gaussianization](#gaussianization)
  - [Journal Articles](#journal-articles)
  - [RBIG](#rbig)
    - [Generalized Divisive Normalization](#generalized-divisive-normalization)

---

## Theory


### Gaussianization

> The original Gaussianization algorithms.

* Gaussianization - Chen & Gopinath - (2000) - [PDF](https://papers.nips.cc/paper/1856-gaussianization.pdf)
* Nonlinear Extraction of 'Independent Components' of elliptically symmetric densities using radial Gaussianization - Lyu & Simoncelli - Technical Report (2008) - [PDF](https://www.cns.nyu.edu/pub/lcv/lyu08a.pdf)


**Applications**

* Gaussianization for fast and accurate inference from cosmological data - Schuhman et. al. - (2016) - [PDF](https://papers.nips.cc/paper/1856-gaussianization.pdf)
* Estimating Information in Earth Data Cubes - Johnson et. al. - EGU 2018
* Multivariate Gaussianization in Earth and Climate Sciences - Johnson et. al. - Climate Informatics 2019 - [repo](https://github.com/IPL-UV/2019_ci_rbig)
* Climate Model Intercomparison with Multivariate Information Theoretic Measures - Johnson et. al. - AGU 2019 - [slides](https://docs.google.com/presentation/d/18KfmAbaJI49EycNule8vvfR1dd0W6VDwrGCXjzZe5oE/edit?usp=sharing)
* Information theory measures and RBIG for Spatial-Temporal Data analysis - Johnson et. al. - In progress


### Journal Articles

* Iterative Gaussianization: from ICA to Random Rotations - Laparra et. al. (2011) - IEEE Transactions on Neural Networks



### RBIG

> The most updated Gaussianization algorithm which generalizes the original algorithms. It is more computationally efficient and still very simple to implement.

* Iterative Gaussianization: from ICA to Random Rotations - Laparra et. al. - IEEE TNNs - [Paper](https://arxiv.org/abs/1602.00229)

**Applications**



---

#### Generalized Divisive Normalization

This can be thought of as an approximation to the Gaussianization 

* Density Modeling of Images Using a Generalized Normalized Transformation - Balle et al. - ICLR (2016) - [arxiv](https://arxiv.org/abs/1511.06281) | [Lecture](http://videolectures.net/iclr2016_balle_density_modeling/) | [PyTorch](https://github.com/jorge-pessoa/pytorch-gdn) | [TensorFlow](https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py)
* End-to-end Optimized Image Compression - Balle et. al. - CVPR (2017) - [axriv](https://arxiv.org/abs/1611.01704)
* Learning Divisive Normalization in Primary Visual Cortex - Gunthner et. al. - [bioarxiv](https://www.biorxiv.org/content/10.1101/767285v1)