# Inference





### Laplace Approximation



This is where we approximate the posterior with a Gaussian distribution $\mathcal{N}(\mu, A^{-1})$.

* $w=w_{map}$, finds a mode (local max) of $p(w|D)$
* $A = \nabla\nabla \log p(D|w) p(w)$ - very expensive calculation
* Only captures a single mode and discards the probability mass 
  * similar to the KLD in one direction.



## Markov Chain Monte Carlo



We can produce samples from the exact posterior by defining a specific Monte Carlo chain.

We actually do this in practice with NNs because of the stochastic training regimes. We modify the SGD algorithm to define a scalable MCMC sampler.



## Variational Inference

**Definition**: We can find the best approximation within a given family w.r.t. KL-Divergence.
$$
\text{KLD}[q||p] = \int_w q(w) \log \frac{q(w)}{p(w|D)}dw 
$$
Let $q(w)=\mathcal{N}(\mu, S)$ and then we minimize KLD$(q||p)$ to find the parameters $\mu, S$.

> "Approximate the posterior, not the model" - James Hensman.