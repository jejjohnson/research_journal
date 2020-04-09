# Entropy


- [Intuition](#intuition)
- [Formulas](#formulas)
  - [Code - Step-by-Step](#code---step-by-step)
  - [Code - Refactored](#code---refactored)
- [Estimating Entropy](#estimating-entropy)
  - [Histogram](#histogram)
  - [Kernel Density Estimation](#kernel-density-estimation)
    - [KNN Approximation](#knn-approximation)
- [Single Variable](#single-variable)
- [Multivariate](#multivariate)
- [Relative Entropy (KL-Divergence)](#relative-entropy-kl-divergence)

## Intuition

> Expected uncertainty.

$$H(X) = \log \frac{\text{\# of Outcomes}}{\text{States}}$$

* Lower bound on the number of bits needed to represent a RV, e.g. a RV that has a unform distribution over 32 outcomes.
  * Lower bound on the average length of the shortest description of $X$
* Self-Information

## Formulas

$$H(\mathbf{X}) = - \int_\mathcal{X} p(\mathbf{x}) \log p(\mathbf{x}) d\mathbf{x}$$

And we can estimate this empirically by:

$$H(\mathbf{X}) = -\sum_{i=1}^N p_i \log p_i$$

where $p_i = P(\mathbf{X})$.

### Code - Step-by-Step

```python
# 1. obtain all possible occurrences of the outcomes
values, counts = np.unique(labels, return_counts=True)

# 2. Normalize the occurrences to obtain a probability distribution 
counts /= counts.sum()

# 3. Calculate the entropy using the formula above
H = - (counts * np.log(counts, 2)).sum()
```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.


### Code - Refactored

```python
# 1. obtain all possible occurrences of the outcomes
values, counts = np.unique(labels, return_counts=True)

# 2. Normalize the occurrences to obtain a probability distribution 
counts /= counts.sum()

# 3. Calculate the entropy using the formula above
base = 2
H = entropy(counts, base=base)
```



## Estimating Entropy


### Histogram


```python
import numpy as np
from scipy import stats

# data
s1 = np.random.normal(10, 10, 1_000)

# construct histogram
hist_pdf, hist_bins = np.histogram(data, bins=50, range=(), density=True)

# calculate the entropy
H_data = stats.entropy(hist_pdf, base=2)
```


### Kernel Density Estimation


#### KNN Approximation





## Single Variable

$$H(X) = \mathbb{E}_{p(X)} \left( \log \frac{1}{p(X)}\right)$$






## Multivariate

$$H(X) = \mathbb{E}_{p(X,Y)} \left( \log \frac{1}{p(X,Y)}\right)$$

## Relative Entropy (KL-Divergence)

Measure of distance between two distributions

$$D_{KL} (P,Q) = \int_\mathcal{X} p(x) \:\log \frac{p(x)}{q(x)}\;dx$$

* aka expected log-likelihood ratio
* measure of inefficiency of assuming that the distribution is $q$ when we know the true distribution is $p$.