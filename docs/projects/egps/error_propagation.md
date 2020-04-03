# Error Propagation

- [Taylor Series Expansion](#taylor-series-expansion)
- [Law of Error Propagation](#law-of-error-propagation)
    - [<font color="red">Proof:</font> Mean Function](#font-color%22red%22prooffont-mean-function)
    - [<font color="red">Proof:</font> Variance Function](#font-color%22red%22prooffont-variance-function)
    - [Resources](#resources)


## Taylor Series Expansion

> A Taylor series is representation of a function as an infinite sum of terms that are calculated from the values of the functions derivatives at a single point - Wiki

Often times we come across functions that are very difficult to compute analytically. Below we have the simple first-order Taylor series approximation.

Let's take some function $f(\mathbf x)$ where $\mathbf{x} \sim \mathcal{N}(\mu_\mathbf{x}, \Sigma_\mathbf{x})$ described by a mean $\mu_\mathbf{x}$ and covariance $\Sigma_\mathbf{x}$. The Taylor series expansion around the function $f(\mathbf x)$ is:

$$\mathbf z = f(\mathbf x) \approx f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) $$

## Law of Error Propagation


This results in a mean and error covariance of the new distribution $\mathbf z$ defined by:

$$\mu_{\mathbf z} = f(\mu_{\mathbf x})$$
$$\Sigma_\mathbf{z} = \nabla_\mathbf{x} f(\mu_{\mathbf x}) \; \Sigma_\mathbf{x} \; \nabla_\mathbf{x} f(\mu_{\mathbf x})^{\top}$$


#### <font color="red">Proof:</font> Mean Function

Given the mean function:

$$\mathbb{E}[\mathbf{x}] = \frac{1}{N} \sum_{i=1} x_i$$

We can simply apply this to the first-order Taylor series function.

$$
\begin{aligned}
\mu_\mathbf{z} &= 
\mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= \mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) \right] +   \mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= f(\mu_{\mathbf x}) + 
\mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mathbf x  \right]- \mathbb{E}_{\mathbf{x}} \left[ \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x} \right] \\
&= f(\mu_{\mathbf x}) +
 \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mu_\mathbf{x} -  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x}  \\
&= f(\mu_{\mathbf x}) \\
\end{aligned}
$$


#### <font color="red">Proof:</font> Variance Function

Given the variance function 

$$\mathbb{V}[\mathbf{x}] = \mathbb{E}\left[ \mathbf{x} - \mu_\mathbf{x} \right]^2$$

$$
\begin{aligned}
\sigma_\mathbf{z}^2
&=
\mathbb{E} \left[ f(\mu_\mathbf{x}) - \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} (\mathbf{x} - \mu_\mathbf{x}) - \mu_\mathbf{x} \right] \\
&=
\mathbb{E} \left[ \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}}  (\mathbf{x} - \mu_\mathbf{x})\right]^2 \\
&=
\left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \mathbb{E}\left[  \mathbf{x} - \mu_\mathbf{x}\right]^2\\
&= \left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \Sigma_\mathbf{x}
\end{aligned}
$$

I've linked a nice tutorial for propagating variances below if you would like to go through the derivations yourself.

---

#### Resources

* Essence of Calculus, Chapter 11 | Taylor Series - 3Blue1Brown - [youtube](https://youtu.be/3d6DsjIBzJ4)
* Introduction to Error Propagation: Derivation, Meaning and Examples - [PDF](http://srl.informatik.uni-freiburg.de/papers/arrasTR98.pdf)
* Statistical uncertainty and error propagation - Vermeer - [PDF](https://users.aalto.fi/~mvermeer/uncertainty.pdf)
