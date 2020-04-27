# Continuous Mixture CDFs

* Author: J. Emmanuel Johnson
* Paper: [Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design](https://arxiv.org/pdf/1902.00275.pdf) - Ho et. al. (2019)


We take K Logistics.

$$x \rightarrow \sigma^{-1}\left[ \text{MixLogCDF}_\theta(x) \right] \cdot \exp(a) + b, $$

where $\theta=[\pi, \mu, \beta]$ are the mixture params.

$$\text{MixLogCDF}_\theta(x) = \sum_{i=1}^K \pi_i \sigma((x-\mu_i) \cdot \exp(-\beta_i))$$

**Domain**

$$\sigma^{-1}(p) \rightarrow \alpha \in \mathbf{R}^{+}, p\in \mathcal{U}([0,1])$$

**CDF Function**

$$F_\theta(x) = \sigma^{-1}\left( \sum_{j=1}^K \pi_j \sigma(\frac{(x-\mu_i)}{\beta_j} \right) $$

**Source**: Flow++

---

### Code Structure

**Forward Transform**

1. Mixture Log CDF(x) 
2. Logit Function
3. Mixture Log PDF

**Inverse Transformation**

1. Sigmoid Function
2. Mixture Inverse CDF
3. Mixture Log PDF

---

### Mixture of Logistics Coupling Layer


---

## Resources


* [Flow++ Model](https://github.com/AlexanderMath/nflow/blob/master/flowpp/models/flowplusplus/log_dist.py)
  > Implementation with a Logistic Mixture Layer. Features the forward and backwards transformation with a bisection search. Uses PyTorch.
* [Flow-Demos](https://github.com/alexlioralexli/flow-demos/blob/4a050ff351a144eef20ccc62b1e2313af8c7f354/deepul_helper/demo1.py) | [Composition Flows](https://github.com/alexlioralexli/flow-demos/blob/4a050ff351a144eef20ccc62b1e2313af8c7f354/deepul_helper/demo2.py)
  > Good Demo showing a basic CDF Flow Model. Also shows the composite flows. However, there is no inverse function.
* [DPP Code](https://github.com/shchur/ifl-tpp/blob/master/code/dpp/flows/logistic_mixture.py)
  > Same as above but a better structure in my opinion.
* [Gaussian Mixture CDF - Jax](https://github.com/Information-Fusion-Lab-Umass/NoX/blob/master/nox/normalizing_flows.py#L1434)
  > Jax Implementation. No inverse but at least I can see the potential Jax version
* [Gaussian Mixture Model - INN 4 Inverse Problems](https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/gaussian_mixture.py) | [Tests](https://github.com/VLL-HD/FrEIA/blob/master/tests/gaussian_mixture.py) | [Technical Report](https://arxiv.org/abs/2003.05739)