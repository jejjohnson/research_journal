# Overview

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Repo: [github.com/jejjohnson/uncertain_gps](https://github.com/jejjohnson/uncertain_gps)



We will do a quick overview to show how we can account for input errors in Gaussian process regression models.


## Problem Statement


**Standard**

$$y = f(\mathbf{x}) + \epsilon_y$$

where $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2)$. Let $\mathbf{x} = \mu_\mathbf{x} + \Sigma_\mathbf{x}$.

**Observe Noisy Estimates**

$$y = f(\mathbf{x}) + \epsilon_y$$

**Observation Means only**



$$y = f(\mu_\mathbf{x}) + \epsilon_y$$

### Posterior Predictions

$$\mu(\mathbf{x}_*) = {\bf k}_* ({\bf K}+\lambda {\bf I}_N)^{-1}{\bf y} = {\bf k}_* \alpha$$

$$\nu_{GP*}^2 =  \sigma_y^2 + k_{**}- {\bf k}_* ({\bf K}+\sigma_y^2 \mathbf{I}_N )^{-1} {\bf k}_{*}^{\top}$$



## Linearized Approximation

Where we take the Taylor expansion of the predictive mean and variance function. The mean function stays the same:

$$\mu_{eGP*}(\mathbf{x_*}) = {\bf k}_* \alpha$$

but the predictive variance term gets changed slightly:

$$\nu_{eGP*}^2 =  
\nu_{GP*}^2(\mathbf{x}_*)  +
{\nabla_{\mu_*}\Sigma_x\nabla_{\mu_*}^\top} +
\text{Tr} \left\{  
  \frac{\partial^2 \nu^2_{GP*}(\mathbf{x_*})}{\partial\mathbf{x_*}\partial\mathbf{x_*}^\top} \bigg\vert_{\mathbf{x_*}=\mu_{\mathbf{x_*}}} \Sigma_\mathbf{x_*}
\right\}
$$

with the term in <font color="red">red</font> being the derivative of the predictive mean function multiplied by the variance.

**Notes**:
* Assumes known variance
* Assumes $D\times D$ covariance matrix for multidimensional data
* Quite inexpensive to implement
* The 3rd term (the 2nd order component of the Taylor expansion) has been show to not make a huge difference

```python
egp_moment1 = jax.jfwd(posterior, args_num=(None, 0))
```

```python
egp_moment2 = jax.hessian(posterior, args_num=(None, 0))
```


## Moment-Matching


**Mean Predictions**

$$\mu_{eGP*}(\mathbf{x_*}) = \mathbf{q}^\top \alpha$$

where:

$$q_i =
|\Lambda^{-1} \Sigma_\mathbf{x_*} + \mathbf{I}|^{-1/2}
\exp\left[ 
  -\frac{1}{2}(\mu_* - \mathbf{x}_i)
  (\Sigma_\mathbf{x_*}+\Lambda)^{-1}
  (\mu_* - \mathbf{x}_j)
\right]
$$

**Variance Predictions**

$$\nu_{eGP*}^2 $$



## Variational

Assumes we have a variational distribution function

$$\mathcal{L}(\theta) = \text{D}_{\text{KL}}\left[ q(\mathbf{f})\, q(\mathbf{X}) || p(\mathbf{f|X})\, p(\mathbf{X}) \right]$$


---

## Other Resources


[**Gaussian Process Model Zoo**](https://jejjohnson.github.io/gp_model_zoo/#/)


---

## Datasets

We use some toy datasets which including: 

1. "near square sine wave"

$$
f(x) = \sin\left(\frac{\pi}{c} \cos\left( 5 + \frac{x}{2} \right)  \right)
$$

2. The sigmoid curve

$$
f(x) = \frac{1}{1+ \exp(-x)}
$$

3. [Mauna Loa Ice Core Data](https://bwengals.github.io/mauna-loa-example-2-ice-core-data.html) | [Data Portal](https://geosci.uchicago.edu/~rtp1/PrinciplesPlanetaryClimate/Data/dataPortal.html)

$$

$$

4. Spatial IASI Data

