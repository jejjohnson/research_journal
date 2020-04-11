# Gaussian Process Basics

- [Data](#data)
- [Model](#model)
- [Posterior](#posterior)
  - [Deterministic Inputs](#deterministic-inputs)
  - [Probabilistic Inputs](#probabilistic-inputs)
- [Variational GP Models](#variational-gp-models)
- [Sparse GP Models](#sparse-gp-models)

## Data

Let's consider that we have the following relationship.

$$y = f(\mathbf{x}) + \epsilon_y$$

Let's assume we have inputs with an additive noise term $\epsilon_y$ and let's assume that it is Gaussian distributed, $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2)$. In this setting, we are not considering any input noise.

---

## Model

Given some training data $\mathbf{X},y$, we are interested in the Bayesian formulation:

$$p(f| \mathbf{X},y) = \frac{{\color{blue}{
p(y| f, \mathbf{X}) } \,\color{darkgreen} {p(f)}}}{\color{red}{
p(y|  \mathbf{X}) }}$$

where we have:

* **GP Prior**, ${\color{darkgreen} {p(f) = \mathcal{GP}(m, k)}}$

We specify a mean function, $m$ and a covariance function $k$.

* **Likelihood**, ${\color{blue}{
p(y| f, \mathbf{X}) = \mathcal{N}(f(\mathbf{X}), \sigma_y^2\mathbf{I})
}}$

which describes the dataset

* **Marginal Likelihood**, ${\color{red}{
p(y|  \mathbf{X}) = \int_f p(y|f, \mathbf{X}) \, p(f|\mathbf{X}) \, df
}}$

* **Posterior**, $p(f| \mathbf{X},y) = \mathcal{GP}(\mu_\text{GP}, \nu^2_\text{GP})$

And the predictive functions $\mu_{GP}$ and $\nu^2_{GP}$ are:

$$
\begin{aligned}
    \mu_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}) \, \mathbf{K}_{GP}^{-1}y=k(\mathbf{x_*}) \, \alpha \\
    \nu^2_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}, \mathbf{x_*}) - k(\mathbf{x_*}) \,\mathbf{K}_{GP}^{-1} \, k(\mathbf{x_*})^{\top}
\end{aligned}
$$

where $\mathbf{K}_\text{GP}=k(\mathbf{x,x}) + \sigma_y^2 \mathbf{I}$.

---

## Posterior

First, let's look at the joint distribution:

$$p(\mathbf{X,Y,F}) $$

### Deterministic Inputs

In this integral, we don't need to propagate a distribution through the GP function. So it should be the standard and we only have to integrate our the function f and condition on our inputs $\mathbf{X}$.

$$
\begin{aligned}
p(\mathbf{Y|X}) &= \int_f p(\mathbf{Y,F|X})\,df \\
&= \int_f p(\mathbf{Y|F}) \, p(\mathbf{F|X})\, df
\end{aligned}
$$

This is a known quantity where we have a closed-form solution to this:

$$p(\mathbf{Y|X}) = \mathcal{N}(\mathbf{Y}|\mathbf{0}, \mathbf{K}+ \sigma_y^2 \mathbf{I})$$

### Probabilistic Inputs

In this integral, we can no longer condition on the $X$'s as they have a probabilistic function. So now we need to integrate them out in addition to the $f$'s.

$$
\begin{aligned}
p(\mathbf{Y}) &= \int_f p(\mathbf{Y,F,X})\,df \\
&= \int_f p(\mathbf{Y|F}) \, p(\mathbf{F|X})\, p(\mathbf{X}) \, df
\end{aligned}
$$


## Variational GP Models


**Posterior Distribution:**
$$p(\mathbf{Y|X}) = \int_{\mathcal F} p(\mathbf{Y|F}) p(\mathbf{F|X}) d\mathbf{F}$$

**Derive the Lower Bound** (w/ Jensens Inequality):

$$\log p(Y|X) = \log \int_{\mathcal F} p(Y|F) P(F|X) dF$$

**importance sampling/identity trick**

$$ = \log \int_{\mathcal F} p(Y|F) P(F|X) \frac{q(F)}{q(F)}dF$$

**rearrange to isolate**: $p(Y|F)$ and shorten notation to $\langle \cdot \rangle_{q(F)}$.

$$= \log \left\langle  \frac{p(Y|F)p(F|X)}{q(F)} \right\rangle_{q(F)}$$

**Jensens inequality**

$$\geq \left\langle \log \frac{p(Y|F)p(F|X)}{q(F)} \right\rangle_{q(F)}$$

**Split the logs**


$$\geq \left\langle \log p(Y|F) + \log \frac{p(F|X)}{q(F)} \right\rangle_{q(F)}$$

**collect terms**

$$\mathcal{L}_{1}(q)=\left\langle \log p(Y|F)\right\rangle_{q(F)} - D_{KL} \left( q(F) || p(F|X)\right) $$

---

## Sparse GP Models

Let's build up the GP model from the variational inference perspective. We have the same GP prior as the standard GP regression model:

$$\mathcal{P}(f) \sim \mathcal{GP}\left(\mathbf m_\theta, \mathbf K_\theta  \right)$$

We have the same GP likelihood which stems from the relationship between the inputs and the outputs:

$$y = f(\mathbf x) + \epsilon_y$$

$$p(y|f, \mathbf{x}) = \prod_{i=1}^{N}\mathcal{P}\left(y_i| f(\mathbf x_i) \right) \sim \mathcal{N}(f, \sigma_y^2\mathbf I)$$

Now we just need an variational approximation to the GP posterior:

$$q(f(\cdot)) = \mathcal{GP}\left( \mu_\text{GP}(\cdot), \nu^2_\text{GP}(\cdot, \cdot) \right) $$

where $q(f) \approx  \mathcal{P}(f|y, \mathbf X)$.



$\mu$ and $\nu^2$ are functions that depend on the augmented space $\mathbf Z$ and possibly other parameters. Now, we can actually choose any $\mu$ and $\nu^2$ that we want. Typically people pick this to be Gaussian distributed which is augmented by some variable space $\mathcal{Z}$ with kernel functions to move us between spaces by a joint distribution; for example:

$$\mu(\mathbf x) = \mathbf k(\mathbf{x, Z})\mathbf{k(Z,Z)}^{-1}\mathbf m$$
$$\nu^2(\mathbf x) = \mathbf k(\mathbf{x,x}) - \mathbf k(\mathbf{x, Z})\left( \mathbf{k(Z,Z)}^{-1}  - \mathbf{k(Z,Z)}^{-1} \Sigma \mathbf{k(Z,Z)}^{-1}\right)\mathbf k(\mathbf{x, Z})^{-1}$$

where $\theta = \{ \mathbf{Z, m_z, \Sigma_z} \}$ are all variational parameters. This formulation above is just the end result of using augmented values by using variational compression (see [here]() for more details). In the end, all of these variables can be adjusted to reduce the KL divergence criteria KL$\left[ q(f)||\mathcal{P}(f|y, \mathbf X)\right]$.

There are some advantages to the approximate-prior approach for example:

* The approximation is non-parametric and mimics the true posterior.
* As the number of inducing points grow, we arrive closer to the real distribution
* The pseudo-points $\mathbf Z$ and the amount are also parameters which can protect us from overfitting.
* The predictions are clear as we just need to evaluate the approximate GP posterior.

**Sources**

* Sparse GPs: Approximate the Posterior, Not the Model - James Hensman (2017) - [blog](https://www.prowler.io/blog/sparse-gps-approximate-the-posterior-not-the-model)