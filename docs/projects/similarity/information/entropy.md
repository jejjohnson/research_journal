# Entropy

- [Intuition](#intuition)
- [Single Variable](#single-variable)
  - [Code - Step-by-Step](#code---step-by-step)
- [Multivariate](#multivariate)
- [Relative Entropy (KL-Divergence)](#relative-entropy-kl-divergence)


## Intuition

> Expected uncertainty.

$$H(X) = \log \frac{\text{\# of Outcomes}}{\text{States}}$$

* Lower bound on the number of bits needed to represent a RV, e.g. a RV that has a unform distribution over 32 outcomes.
  * Lower bound on the average length of the shortest description of $X$
* Self-Information

## Single Variable

$$H(X) = \mathbb{E}_{p(X)} \left( \log \frac{1}{p(X)}\right)$$


<details>
<summary>Code - From Scratch</summary>

### Code - Step-by-Step

1. Obtain all of the possible occurrences of the outcomes. 
   ```python
   values, counts = np.unique(labels, return_counts=True)
   ```

2. Normalize the occurrences to obtain a probability distribution
   ```python
   counts /= counts.sum()
   ```

3. Calculate the entropy using the formula above
   ```python
   H = - (counts * np.log(counts, 2)).sum()
   ```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.
</details>


<details>
<summary>Code - Refactor</summary>

2. Use Scipy Function
   ```python
   H = entropy(counts, base=base)
   ```
</details>

## Multivariate

$$H(X) = \mathbb{E}_{p(X,Y)} \left( \log \frac{1}{p(X,Y)}\right)$$

## Relative Entropy (KL-Divergence)

Measure of distance between two distributions

$$D_{KL} (P,Q) = \int_\mathcal{X} p(x) \:\log \frac{p(x)}{q(x)}\;dx$$

* aka expected log-likelihood ratio
* measure of inefficiency of assuming that the distribution is $q$ when we know the true distribution is $p$.