# Coupling Layers

> A univariate bijective differentiable function $\hat{f}_\theta(x): \mathbb{R} \rightarrow \mathbb{R}$, parameterized by $\theta$. **Note**: It needs to be strictly monotonic.

---

### Non-Linear Squared Flows

$$\hat{f}_\theta(x) = ax + b + \frac{c}{1+(dx+h)^2}$$

where $\theta=[a,b,c,d,h]$

* Invertible
* Inverse is analytically computable (root of cubic polynomial)

**Paper**: Latent Normalizing Flows for Discrete Sequences - Ziegler & Rush (2019)

---

### Continuous Mixture CDFs


$$\hat{f}_\theta(x) = \theta_1 F_{\theta_3}(x) + \theta_2$$

where $\theta_1 \neq \theta, \theta_3\in \mathbb{R}, \theta_2=[\pi, \mu, \sigma]\in \mathbb{R}^K \times \mathbb{R}^K \times \mathbb{R}^K$

The function $F_{\theta_2}(x, \pi, \mu, \sigma)$ is a CDF mixture distribution of $K$ logistic functions, post-composed with an inverse Sigmoid function, logit $= \log p / (1-p)$. So the full function is:

$$F(x, \pi, \mu, \sigma) = \text{logit}\left( \sum_{j=1}^{K} \pi_j \text{ logistic}\left( \frac{x-\mu_j}{\sigma}  \right) \right)$$

Some notes:
* $\text{logit}:[0,1] \rightarrow \mathbb{R}$ - ensure the right range for $\hat{f}$
* Inverse: done numerically w/ the bisection algorithm
* $\nabla_x F(\cdot)$ - it's a mixture of PDFs of logistic mixture distribution (i.e. linear combination of hyperbolic secant functions)

**Paper**: Flow++ - Ho et. al. (2019)

---

### Splines

> A spline is a piece-wise polynomial or a piece-rational function which is specified by $K+1$ points $(x_i,y_i)_{i=0}^K$ called knots which a spline is passed.

In particular, I am interested in rational-quadratic splines.

> Models a coupling layer $\hat{f}_\theta(x)$ as a monotone rational-quadratic spline on the interval $[-B, B]$, and outside the interval as an identity function.

**Paper**: Neural Spline Flows, Durkan et. al. (2019)