# Solving Hard Integral Problems

[**Source**](https://www.cs.ubc.ca/~schmidtm/MLRG/GaussianProcesses.pdf) | Deisenroth - [Sampling](https://drive.google.com/file/d/1Ryb1zDzndnv1kOe8nT0Iu4OD6m0KC8ry/view)

**Advances in VI** - [Notebook](https://github.com/magister-informatica-uach/INFO320/blob/master/6_advances_in_VI.ipynb)

* Numerical Integration (low dimension)
* Bayesian Quadrature
* Expectation Propagation
* Conjugate Priors (Gaussian Likelihood w/ GP Prior)
* Subset Methods (Nystrom)
* Fast Linear Algebra (Krylov, Fast Transforms, KD-Trees)
* Variational Methods (Laplace, Mean-Field, Expectation Propagation)
* Monte Carlo Methods (Gibbs, Metropolis-Hashings, Particle Filter)




# Inference

## Maximum Likelihood


**Sources**:

* [Intro to Quantitative Econ w. Python](https://python-intro.quantecon.org/mle.html)



---

## Laplace Approximation



This is where we approximate the posterior with a Gaussian distribution $\mathcal{N}(\mu, A^{-1})$.

* $w=w_{map}$, finds a mode (local max) of $p(w|D)$
* $A = \nabla\nabla \log p(D|w) p(w)$ - very expensive calculation
* Only captures a single mode and discards the probability mass 
  * similar to the KLD in one direction.

---

## Markov Chain Monte Carlo

We can produce samples from the exact posterior by defining a specific Monte Carlo chain.

We actually do this in practice with NNs because of the stochastic training regimes. We modify the SGD algorithm to define a scalable MCMC sampler.

[Here](https://chi-feng.github.io/mcmc-demo/) is a visual demonstration of some popular MCMC samplers.



## Variational Inference

**Definition**: We can find the best approximation within a given family w.r.t. KL-Divergence.
$$
\text{KLD}[q||p] = \int_w q(w) \log \frac{q(w)}{p(w|D)}dw 
$$
Let $q(w)=\mathcal{N}(\mu, S)$ and then we minimize KLD$(q||p)$ to find the parameters $\mu, S$.

> "Approximate the posterior, not the model" - James Hensman.