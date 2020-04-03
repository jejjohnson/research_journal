# Input Uncertainty for Gaussian Processes

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Repo: [github.com/jejjohnson/uncertain_gps](https://github.com/jejjohnson/uncertain_gps)



---

## Problem Statement


**Standard**

$$y = f(\mathbf{x}) + \epsilon_y$$

where $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2)$. Let $\mathbf{x} = \mu_\mathbf{x} + \Sigma_\mathbf{x}$.

**Observe Noisy Estimates**

$$y = f(\mathbf{x}) + \epsilon_y$$

**Observation Means only**



$$y = f(\mu_\mathbf{x}) + \epsilon_y$$




## Main Methods

---

### Linearization

Where we take the Taylor expansion of the predictive mean and variance function. The mean function stays the same:

$$\mu_{GP*} = {\bf k}_* ({\bf K}+\lambda {\bf I}_N)^{-1}{\bf y} = {\bf k}_* \alpha$$

but the predictive variance term gets changed slightly:

$$ \nu_{GP*}^2 = \sigma_y^2 + {\color{red}{\nabla_{\mu_*}\Sigma_x\nabla_{\mu_*}^\top} }+ k_{**}- {\bf k}_* ({\bf K}+\sigma_y^2 \mathbf{I}_N )^{-1} {\bf k}_{*}^{\top}$$

with the term in <font color="red">red</font> being the derivative of the predictive mean function multiplied by the variance.

**Notes**:
* Assumes known variance
* Assumes $D\times D$ covariance matrix for multidimensional data
* Quite inexpensive to implement


---

### Moment-Matching


---

### Variational

Assumes we have a variational distribution function

$$\mathcal{L}(\theta) = \text{D}_{\text{KL}}\left[ q(\mathbf{f})\, q(\mathbf{X}) || p(\mathbf{f|X})\, p(\mathbf{X}) \right]$$


---

## Other Resources


[**Gaussian Process Model Zoo**](https://jejjohnson.github.io/gp_model_zoo/#/)
