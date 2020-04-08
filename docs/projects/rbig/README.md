# Rotation-Based Iterative Gaussianization


A method that provides a transformation scheme from any distribution to a gaussian distribution. This repository will facilitate translating the original MATLAB code into a python implementation compatible with the scikit-learn framework.


### Resources

* Original Webpage - [ISP](http://isp.uv.es/rbig.html)
* Original MATLAB Code - [webpage](http://isp.uv.es/code/featureextraction/RBIG_toolbox.zip)
* Original Python Code - [github](https://github.com/spencerkent/pyRBIG)
* [Paper](https://arxiv.org/abs/1602.00229) - Iterative Gaussianization: from ICA to Random Rotations

Abstract From Paper

> Most signal processing problems involve the challenging task of multidimensional probability density function (PDF) estimation. In this work, we propose a solution to this problem by using a family of Rotation-based Iterative Gaussianization (RBIG) transforms. The general framework consists of the sequential application of a univariate marginal Gaussianization transform followed by an orthonormal transform. The proposed procedure looks for differentiable transforms to a known PDF so that the unknown PDF can be estimated at any point of the original domain. In particular, we aim at a zero mean unit covariance Gaussian for convenience. RBIG is formally similar to classical iterative Projection Pursuit (PP) algorithms. However, we show that, unlike in PP methods, the particular class of rotations used has no special qualitative relevance in this context, since looking for interestingness is not a critical issue for PDF estimation. The key difference is that our approach focuses on the univariate part (marginal Gaussianization) of the problem rather than on the multivariate part (rotation). This difference implies that one may select the most convenient rotation suited to each practical application. The differentiability, invertibility and convergence of RBIG are theoretically and experimentally analyzed. Relation to other methods, such as Radial Gaussianization (RG), one-class support vector domain description (SVDD), and deep neural networks (DNN) is also pointed out. The practical performance of RBIG is successfully illustrated in a number of multidimensional problems such as image synthesis, classification, denoising, and multi-information estimation.


---

## Software

I have spent some time developing a python package to be able to use RBIG for many applications. I tried to break everything up into as many components as possible. It's fairly extensible but within a limited scope. I also tried hard to follow the `scikit-learn` conventions so that the learning curve is as small as possible.

### Installation Instructions


We can pip install the package directly from the repository.

```bash
pip install "git+https://github.com/jejjohnson/rbig.git#egg=rbig"
```

We can create a conda environment and simply install all the packages manually.

```bash
conda env create -f environment.yml
```

### Features

#### Transformations


**Uniformization**
* Scipy Histogram Transformation - The simplest way to do it
* Quantile Transformer - Focuses on estimating the CDF with options for estimating the PDF
* Histogram Transformation (from scratch) - I do my own version which focuses on extending the CDF to allow for outliers
* Mixture of Gaussians - A fully parameterized version 

**Inverse Gaussian CDF**

**Rotations**
* Principal Components Analysis (PCA)
* Random Orthogonal Rotations
* ICA (**TODO**)

#### Univariate Entropy Estimators

* Kernel Density Estimation
* Histogram
* Multivariate Gaussian
* K-Nearest Neighbours


#### Loss Functions

* Maximum Layers
* Information Loss - the change in total correlation
* Neg-Entropy 
* Change in Negentropy
* Negative Log-Likelihood

#### API

* Layers
* Params
* Models
* Losses