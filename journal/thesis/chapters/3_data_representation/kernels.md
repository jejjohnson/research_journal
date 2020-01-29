# Kernel Methods

* Kernel Methods
  * What is a Kernel function?
  * Interpreting Kernel Functions (Parameters)
  * Derivatives + Sensivity
  * Literature
* Gaussian Processes
* Deep Gaussian Processes

---

## 1 Kernel Methods

### 1.1 What is a Kernel Function?

* Similarity Measure + NonLinear Function
* higher order features

**Examples**
* Regression (KRR, GPR)
* Classification (SVM)
* Feature Selection (KPOLS)
* Dimensionality Reduction ()
* Output Normalized Methods (Laplacian Eigenmaps)
* Density Estimation (KDE)
* Dependence Estimation (HSIC)
* Distance (MMD)
* Scaling (Nystrom, rNystrom, RKS)

<details>

* [Blog](http://evelinag.com/Ariadne/covarianceFunctions.html)
* GPML Book - [Chapter 4 - Covariance Functions](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf)
* Gustau - [Kernel Methods in Machine Learning](http://isp.uv.es/courses.html)

**Intuition**
* [From Dependence to Causation](https://arxiv.org/pdf/1607.03300.pdf)
  > Nice introduction with a nice plot. Also discusses the nystrom method randomized kernels. I like like to 'different representations'. Can use example where we can do dot products between features to use a nice linear method.
* [Kernel Methods for Pattern Analysis]()

</details>

---

### 1.2 Interpreting Kernel Parameters


#### Example RBF Kernel

This is by far the most popular kernel you will come across when you are working with kernel methods. It's the most generalizable and it's fairly easy to train and interpret.

$$K(x,y)=\nu^2 \cdot \text{exp}\left( - \frac{d(x,y)}{2\lambda^2} \right) + \sigma_{\text{noise}}^2 \cdot \delta_{ij}$$

where:

* $\nu^2 > 0$ is the signal variance
* $\lambda > 0$ is the length scale
* $\sigma_{\text{noise}}^2$ is the noise variance

**Distance Metric** - $d(x,y)_{E}$

This is the distance metric used to describe the kernel function. For the RBF kernel, this distance metric is the Euclidean distance ($||x-y||_2^2$). But we can switch this measure for another, e.g. the $L_1$-norm produces the Laplacian kernel. In most 'stationary' kernels, this measure is the foundation for this family of kernels. It acts as a smoother.

**Signal Variance - $\nu^2>0$**

This is simply a scaling factor. It determines the variation of function values from their mean. Small values of $\nu^2$ characterize functions that stay close to their mean value, larger values allow for more variation. If the signal variance is too large, the modelled function will be more susceptible to outliers. **Note**: Almost every kernel has this parameter in front as it is just a scale factor.

**Length Scale - $\lambda > 0$**

This parameter describes how smooth a function is. Small $\lambda$ means the function values change quickly, large $\lambda$ characterizes functions that change slowly. This parameter determines how far we can reliably extrapolate from the training data. **Note:** Put $\lambda$ is constant, automatic relevance determine, diagonal, etc.

**Noise Variance - $\sigma_{\text{noise}}^2>0$**

This parameter describes how much noise we expect to be present in the data.

#### Other Kernels

**Issues**:
* Limited Expressiveness
* Neural Networks - Infinitely Wide

**Other Kernels**
* Spectral Mixture Kernel
* ArcCosine - Neural Networks
* Neural Tangent Kernels
* Deep Kernel learning
* Convolutional Kernels

<details>

**Kernel Resources**
* Learning with Spectral Kernels - [Prezi](https://www.hiit.fi/wp-content/uploads/2018/04/Spectral-Kernels-S12.pdf)
* Learning Scalable Deep Kernels with Recurrent Structure - Al Shedivat (2017)
* Deep Kernel Learning - Wilson (2015)
* Deep convolutional Gaussian processes - Blomqvist (2018)
* Stochastic Variational Deep Kernel Learning - Wilson (2016)
* Non-Stationary Spectral Kernels - Remes (2017)

**Resources**
* Kernel Cookbook
* A Visual Comparison of Gaussian Regression Kernels - [towardsdatascience](https://towardsdatascience.com/a-visual-comparison-of-gaussian-process-regression-kernels-8d47f2c9f63c)
* [Kernel Functions](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/)
* [PyKernels](https://github.com/gmum/pykernels)
* [Sklearn kernels](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/gaussian_process/kernels.py#L1146)

**Key Figures**
* [Markus Heinonen](https://users.aalto.fi/~heinom10/)

**Cutting Edge**
* [Bayesian Approaches to Distribution Regression](http://www.gatsby.ucl.ac.uk/~dougals/slides/bdr-nips/#/)
* HSIC
  * [HSIC, A Measure of Independence?](http://www.cmap.polytechnique.fr/~zoltan.szabo/talks/invited_talk/Zoltan_Szabo_invited_talk_EPFL_LIONS_28_02_2018_slides.pdf)
  * [Measuring Dependence and Conditional Dependence with Kernels](http://people.tuebingen.mpg.de/causal-learning/slides/CaMaL_Fukumizu.pdf)

</details>

---

### 3 Derivatives & Sensitivity

<details>

**Resources**
* Computing gradients via GPR - [stack](https://stats.stackexchange.com/questions/373446/computing-gradients-via-gaussian-process-regression)
  > Nice overfew of formula in a very easy way.
* [Derivatives of GP](https://mathoverflow.net/questions/289424/derivatives-of-gaussian-processes)
  > Simple way to take a derivative of a kernel
* [Differentiating GPs](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf)
* [GPyTorch Implementation](https://gpytorch.readthedocs.io/en/latest/examples/10_GP_Regression_Derivative_Information/index.html)



**Literature**
* [Direct Density Derivative and its Applications to KLD](http://proceedings.mlr.press/v38/sasaki15.pdf)
* [Kernel Derivative Approximation with RFF](https://arxiv.org/abs/1810.05207)
* An SVD and Derivative Kernel Approach to Learning from Geometric Data - Wong 
* Scaling Gaussian Process Regression with Derivatives - Eriksson 2018
* MultiDimensional SVM to Include the Samples of the Derivatives in the Reconstruction Function - Perez-Cruz ()
* Linearized Gaussian Processes for Fast Data-driven Model Predictive Control - Nghiem et. al. (2019)
</details>

---

## Bayesian Treatment: Gaussian Processes

---

## Composition: Deep Gaussian Processes

---

## Connections

