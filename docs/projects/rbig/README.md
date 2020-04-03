# Rotation-Based Iterative Gaussianization


A method that provides a transformation scheme from any distribution to a gaussian distribution. This repository will facilitate translating the original MATLAB code into a python implementation compatible with the scikit-learn framework.


### Resources

* Original Webpage - [ISP](http://isp.uv.es/rbig.html)
* Original MATLAB Code - [webpage](http://isp.uv.es/code/featureextraction/RBIG_toolbox.zip)
* Original Python Code - [github](https://github.com/spencerkent/pyRBIG)
* [Paper](https://arxiv.org/abs/1602.00229) - Iterative Gaussianization: from ICA to Random Rotations

Abstract From Paper

> Most signal processing problems involve the challenging task of multidimensional probability density function (PDF) estimation. In this work, we propose a solution to this problem by using a family of Rotation-based Iterative Gaussianization (RBIG) transforms. The general framework consists of the sequential application of a univariate marginal Gaussianization transform followed by an orthonormal transform. The proposed procedure looks for differentiable transforms to a known PDF so that the unknown PDF can be estimated at any point of the original domain. In particular, we aim at a zero mean unit covariance Gaussian for convenience. RBIG is formally similar to classical iterative Projection Pursuit (PP) algorithms. However, we show that, unlike in PP methods, the particular class of rotations used has no special qualitative relevance in this context, since looking for interestingness is not a critical issue for PDF estimation. The key difference is that our approach focuses on the univariate part (marginal Gaussianization) of the problem rather than on the multivariate part (rotation). This difference implies that one may select the most convenient rotation suited to each practical application. The differentiability, invertibility and convergence of RBIG are theoretically and experimentally analyzed. Relation to other methods, such as Radial Gaussianization (RG), one-class support vector domain description (SVDD), and deep neural networks (DNN) is also pointed out. The practical performance of RBIG is successfully illustrated in a number of multidimensional problems such as image synthesis, classification, denoising, and multi-information estimation.