## Gaussian Distributions


### Univariate Gaussian

$$\mathcal{P}(x|\mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left( -\frac{1}{2\sigma^2}(x - \mu)^2 \right)$$

### Multivariate Gaussian

$$\begin{aligned}
\mathcal{P}(x | \mu, \Sigma) &= \mathcal{N}(\mu, \Sigma) \\
&= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\sqrt{\text{det}|\Sigma|}}\text{exp}\left( -\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu) \right)
\end{aligned}$$


### Joint Gaussian Distribution

$$\begin{aligned}\mathcal{P}(x, y) &= 
\mathcal{P}\left(\begin{bmatrix}x \\ y\end{bmatrix} \right) \\
&= \mathcal{N}\left( 
    \begin{bmatrix}
    a \\ b
    \end{bmatrix},
    \begin{bmatrix}
    A & B \\ B^{\top} & C
    \end{bmatrix} \right)
    \end{aligned}$$


### Marginal Distribution $\mathcal{P}(\cdot)$

We have the marginal distribution of $x$

$$\mathcal{P}(x) \sim \mathcal{N}(a, A)$$

and in integral form:

$\mathcal{P}(x) = \int_y \mathcal{P}(x,y)dy$

and we have the marginal distribution of $y$

$$\mathcal{P}(y) \sim \mathcal{N}(b, B)$$

### Conditional Distribution $\mathcal{P}(\cdot | \cdot)$

We have the conditional distribution of $x$  given $y$.

$$\mathcal{P}(x|y) \sim \mathcal{N}(\mu_{a|b}, \Sigma_{a|b})$$

where:

* $\mu_{a|b} = a + BC^{-1}(y-b)$
* $\Sigma_{a|b} = A - BC^{-1}B^T$

and we have the marginal distribution of $y$ given $x$

$$\mathcal{P}(y|x) \sim \mathcal{N}(\mu_{b|a}, \Sigma_{b|a})$$

where:

* $\mu_{b|a} = b + AC^{-1}(x-a)$
* $\Sigma_{b|a} = B - AC^{-1}A^T$

basically mirror opposites of each other. But this might be useful to know later when we deal with trying to find the marginal distributions of Gaussian process functions.

**Source**:

* Sampling from a Normal Distribution - [blog](https://juanitorduz.github.io/multivariate_normal/)
  > A really nice blog with nice plots of joint distributions.
* Two was to derive the conditional distributions - [stack](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution?noredirect=1&lq=1)
* How to generate Gaussian samples = [blog](https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959s)


Multivariate Gaussians and Detereminant - [Lecturee Notes](http://courses.washington.edu/b533/lect4.pdf)


---

### Bandwidth Selection


**Scotts**

```python
sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))
```

**Silverman**

```python
sigma = np.power(n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
```
