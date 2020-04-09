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

