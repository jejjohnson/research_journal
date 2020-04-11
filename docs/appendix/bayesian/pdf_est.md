# PDF Estimation

## Main Idea

<center>

<p align="center">
<img src=projects/rbig/software/pics/input_dist.png" />

<b>Fig I</b>: Input Distribution.
</center>
</p>


$$P(x \in [a,b]) = \int_a^b p(x)dx$$


#### Likelihood

Given a dataset $\mathcal{D} = \{x^{1}, x^{2}, \ldots, x^{n}\}$, we can find the some parameters $\theta$ by solving this optimization function: the likelihood

$$\underset{\theta}{\text{max}} \sum_i \log p_\theta(x^{(i)})$$

or equivalently:

$$\underset{\theta}{\text{min }} \mathbb{E}_x \left[ - \log p_\theta(x) \right]$$

This is equivalent to minimizing the KL-Divergence between the empirical data distribution $\tilde{p}_\text{data}(x)$ and the model $p_\theta$.

$$
D_\text{KL}(\hat{p}(\text{data}) || p_\theta) 
= \mathbb{E}_{x \sim \hat{p}_\text{data}} 
\left[ - \log p_\theta(x) \right] - H(\hat{p}_\text{data})
$$

where $\hat{p}_\text{data}(x) = \frac{1}{n} \sum_{i=1}^N \mathbf{1}[x = x^{(i)}]$

#### Stochastic Gradient Descent

Maximum likelihood is an optimization problem so we can use stochastic gradient descent (SGD) to solve it. This algorithm minimizes the expectation for $f$ assuming it is a differentiable function of $\theta$.

$$\argmin_\theta \mathbb{E} \left[ f(\theta) \right]$$

With maximum likelihood, the optimization problem becomes:

$$\argmin_\theta \mathbb{E}_{x \sim \hat{p}_\text{data}} \left[ - \log p_\theta(x) \right]$$

We typically use SGD because it works with large datasets and it allows us to use deep learning architectures and convenient packages.


---

### Example

#### Mixture of Gaussians

$$p_\theta(x) = \sum_i^k \pi_i \mathcal{N}(x ; \mu_i, \sigma_i^2)$$

where we have parameters as $k$ means, variances and mixture weights,

$$\theta = (\pi_1, \cdots, \pi_k, \mu_1, \cdots, \mu_k, \sigma_1, \cdots, \sigma_k)$$

However, this doesn't really work for high-dimensional datasets. To sample, we pick a cluster center and then add some Gaussian noise.

## Histogram Method


## Gotchas

### Search Sorted


**Numpy**

```python

```

**PyTorch**

```python
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    h_sorted = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    return h_sorted
```

This is an unofficial implementation. There is still some talks in the PyTorch community to implement this. See github issue [here](https://github.com/pytorch/pytorch/issues/1552). For now, we just use the implementation found in various [implementations](https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/spline_flows.py#L20).