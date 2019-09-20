# Hibert-Schmidt Independence Criterion (HSIC)

---
## Resources

**Presentations**

* [HSIC, A Measure of Independence](http://www.cmap.polytechnique.fr/~zoltan.szabo/talks/invited_talk/Zoltan_Szabo_invited_talk_EPFL_LIONS_28_02_2018_slides.pdf) - Szabo (2018)


---
## Practical Equations


**HSIC**

$$
\text{MMD}^2(\hat{P}_{XY}, \hat{P}_X\hat{P}_Y, \mathcal{H}_k) \coloneqq 
\frac{1}{n^2}\text{tr} \left( K_x H K_y H \right)
$$

where $H$ is the centering matrix $H=I_n-\frac{1}{n}1_n1_n^\top$.

$$
\text{HSIC}^2(\hat{P}_{XY}, \mathcal{F}, \mathcal{G}) = 
\text{MMD}^2(\hat{P}_{XY}, \hat{P}_X\hat{P}_Y, \mathcal{H}_k)
$$

$$$$

---
### Objects of Interest


#### Mean Embedding

$$\mu_k(\mathbb{P}) \coloneqq \int_\mathcal{X} \underbrace{\psi(x)}_{k(\cdot,x)} d\mathbb{P}(x) \in \mathcal{H}_k$$

#### Maximum Mean Discrepency (MMD)

$$\text{MMD}_k(\mathbb{P}, \mathbb{Q}) \coloneqq || \mu_k(\mathbb{P}) - \mu_k(\mathbb{Q}) \in ||_{\mathcal{H}_k}$$


#### Hilbert-Schmidt Independence Criterion (HSIC)

$$\text{HSIC}_k(\mathbb{P}) \coloneqq \text{MMD}_k(\mathbb{P}, \otimes_{m=1}^M \mathbb{P}_m)$$



---
### Classical Information Theory

**Kullback-Leibler Divergence**

$$D_{KL}(\mathbb{P}, \mathbb{Q}) = \int_{\mathbb{R}^d} p(x) \log \left[ \frac{p(x)}{q(x)} \right]dx$$

**Mutual Information**

$$I(\mathbb{P})=D_{KL}\left( \mathbb{P}, \otimes_{m=1}^{M}\mathbb{P}_m \right)$$


---
### Tangent Kernel Alignment



**Original Kernel Tangent Alignment**

$$A(K_x, K_y) = 
\frac{\left\langle K_x, K_y \right\rangle_{F}}{\sqrt{|| K_x||_{F}|| K_y ||_{F}}}
$$

The alignment can be seen as a similarity score based on the cosine of the angle. For arbitrary matrices, this score ranges between -1 and 1. But using positive semidefinite Gram matrices, the score is lower-bounded by 0.

**Centered Kernel Tangent Alignment**

$$A(K_{cx}, K_{cy}) = 
\frac{\left\langle K_{cx}, K_{cy} \right\rangle_{F}}{\sqrt{|| K_{cx}||_{F}|| K_{cy} ||_{F}}}
$$

They add a normalization term to deal with some of the shortcomings of the original KTA algorithm which had some benefits e.g. a way to cancel out unbalanced class effects. The improvement over the original algorithm seems minor but there is a critical difference. Without the centering, the alignment does not correlate well to the performance of the learning machine. 


---
### Covariance vs Correlation

The above methods can be put into perspective of the difference between covariance measures and correlation meausres. In this section, I will explain why that is so. We can write out the full definitions for covariance and correlation. The definition of covariance is:

$$\text{cov}_{XY}=\sigma_{XY}=E\left[(X - \mu_X)(Y - \mu_Y) \right]$$

The definition of correlation is:

$$\text{corr}_{XY}=\rho_{XY}=\frac{\sigma_{XY}}{\sigma_X \sigma_Y}$$

The correlation measure is dimensionless whereas the covariance is in units obtained by multiplying the units of the two variables.

**Source**: 

* [Wikipedia](https://en.wikipedia.org/wiki/Covariance_and_correlation) on Covariance and correlation
* [Wikipedia](https://en.wikipedia.org/wiki/Correlation_and_dependence) on Correlation and dependence


---
## Literature Review


**An Overview of Kernel Alignment and its Applications** - Wang et al (2012) - [PDF](https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf)

This goes over the literature of the kernel alignment method as well as some applications it has been used it.


### Applications

* Kerneel Target Alignment Parameter: A New Modelability for Regression Tasks - Marcou et al (2016) - [Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00539)
* Brain Activity Patterns - [Paper](https://www.frontiersin.org/articles/10.3389/fnins.2017.00550/full)
* Scaling - [Paper](https://link.springer.com/article/10.1007/s11222-016-9721-7)


### Textbooks

* Kernel Methods for Digital Processing - [Book](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118705810)
