# Uncertainty of GPs: Taylor Expansion

In this document, I will be showing how we can use the Taylor Expansion approach to the posterior of Gaussian process algorithm

## GP Model

We have a standard GP model.
$$
\begin{aligned}
y &= f(x) + \epsilon_y\\
\epsilon_y &\sim \mathcal{N}(0, \sigma_y^2) \\
\end{aligned}
$$

* **GP Prior**: $p(f|X)\sim\mathcal{N}(m(X), \mathbf{K})$
* **Gaussian Likelihood**: $p(y|f, X)=\mathcal{N}(y|f(x), \sigma_y^2\mathbf{I})$
* **Posterior**: $f \sim \mathcal{GP}(f|\mu_{GP}, \nu^2_{GP})$ 

And the predictive functions $\mu_{GP}$ and $\nu^2_{GP}$ are:
$$
\begin{align}
    \mu_{GP} &= K_{*} K_{GP}^{-1}y=K_{*} \alpha \\
    \nu^2_{GP} &= K_{**} - K_{*} K_{GP}^{-1}K_{*}^{\top}
\end{align}
$$


## Taylor Expansion

Let's assume we have inputs with an additive noise term $\epsilon_x$ and let's assume that it is Gaussian distributed. We can write some expressions which are very similar to the GP model equations specified above:
$$
\begin{aligned}
y &= f(x) + \epsilon_y \\
x &= \mu_x + \epsilon_x \\
\epsilon_y &\sim \mathcal{N} (0, \sigma_y^2) \\
\epsilon_x &\sim \mathcal{N} (0, \Sigma_x)
\end{aligned}
$$
This is the transformation of a Gaussian random variable $x$ through another r.v. $y$ where we have some additive noise $\epsilon_y$. The biggest difference is that the GP model assumes that $x$ is deterministic whereas we assume here that $x$ is a random variable itself. Because we know that integrating out the $x$'s is quite difficult to do in practice (because of the nonlinear Kernel functions), we can make an approximation of $f(\cdot)$ via the Taylor expansion. We can take the a 2nd order Taylor expansion of $f$ to be:
$$
\begin{aligned}
f(x) &\approx f(\mu_x + \epsilon_x) \\
     &\approx f(\mu_x) + \nabla f(\mu_x) \epsilon_x 
     + \sum_{i}\frac{1}{2} \epsilon_x^\top \nabla^2 f(\mu_x) \epsilon_x 
\end{aligned}
$$
where $\nabla_x$ is the gradient of the function $f(\mu_x)$ w.r.t. $x$ and $\nabla_x^2 f(\mu_x)$ is the second derivative (the Hessian) of the function $f(\mu_x)$ w.r.t. $x$. This is a second-order approximation which has that expensive Hessian term. There have have been studies that have shown that that term tends to be neglible in practice and a first-order approximation is typically enough. Now the question is: where to put use the Taylor expansion within the GP model? There are two options: the model or the posterior. We will outline the two approaches below.



## Approximate the Model



## Approximate The Posterior





We can compute the expectation $\mathbb{E}[\cdot]$ and variance $\mathbb{V}[\cdot]$ of this Taylor expansion to come up with an approximate mean and variance function for our posterior.

### Expectation

This calculation is straight-forward because we are taking the expected value of a mean function $f(\mu_x)$, the derivative of a mean function $f(\mu_x)$ and a Gaussian distribution noise term $\epsilon_x$ with mean 0. 
$$
\begin{aligned}
\mathbb{E}[f(x)] &\approx \mathbb{E}[f(\mu_x) + \nabla f(\mu_x) \epsilon_x] \\
								 &= f(\mu_x) + \nabla f(\mu_x) \mathbb{\epsilon_x} \\
								 &= f(\mu_x)
\end{aligned}
$$

### Variance

The variance term is a bit more complex.
$$
\begin{aligned}
\mathbb{E}\left[(f(x) - \mathbb{E}[f(x)])^\top(f(x) - \mathbb{E}[f(x)])\right] 
			&\approx \mathbb{E}\left[(f(x) - f(\mu_x))^\top(f(x) - f(\mu_x))\right] \\
			&\approx \mathbb{E} \left[ \left(f(\mu_x) + \nabla f(\mu_x)\epsilon_x \right) 
			\left( f(\mu_x) + \nabla f(\mu_x)\epsilon_x\right)^\top\right] \\
			&= \mathbb{E} \left[ \left(\nabla f(\mu_x)\: \epsilon_x  \right)^\top
			\left( \nabla f(\mu_x)\: \epsilon_x \right) \right] \\
			&= \nabla f(\mu_x) \mathbb{E}[\epsilon_x\epsilon_x^\top]\nabla f(\mu_x) \\
			&= \nabla f(\mu_x)\: \Sigma_x \:\nabla f(\mu_x)
\end{aligned}
$$

### I: Additive Noise Model ($x,f$)

This is the noise
$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_{y} 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_x & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$
where
$$
\begin{aligned}
\mu_y &= f(\mu_x) \\
\Pi &= \nabla_x f(\mu_x) \: \Sigma_x \: \nabla_x f(\mu_x)^\top + \nu^2(x) \\
C &= \Sigma_x \: \nabla_x^\top f(\mu_x)
\end{aligned}
$$
So if we want to make predictions with our new model, we will have the final equation as:
$$
\begin{align}
f &\sim \mathcal{N}(f|\mu_{GP}, \nu^2_{GP}) \\
    \mu_{GP} &= K_{*} K_{GP}^{-1}y=K_{*} \alpha \\
    \nu^2_{GP} &= K_{**} - K_{*} K_{GP}^{-1}K_{*}^{\top} + \tilde{\Sigma}_x
\end{align}
$$
where $\tilde{\Sigma}_x = \nabla_x \mu_{GP} \Sigma_x \nabla \mu_{GP}^\top$.



##### Other GP Methods

We can extend this method to other GP algorithms including sparse GP models. The only thing that changes are the original $\mu_{GP}$ and $\nu^2_{GP}$ equations. In a sparse GP we have the following predictive functions
$$
\begin{align}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top}
\end{align}
$$
So the new predictive functions will be:
$$
\begin{align}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top} 
    + \tilde{\Sigma}_x
\end{align}
$$
As shown above, this is a fairly extensible method that offers a cheap improved predictive variance estimates on an already trained GP model. Some future work could be evaluating how other GP models, e.g. Sparse Spectrum GP, Multi-Output GPs, e.t.c.

### II: Non-Additive Noise Model



### III: Quadratic Approximation





## Parallels to the Kalman Filter

The Kalman Filter (KF) community use this exact formulation to motivate the Extended Kalman Filter (EKF) algorithm and some variants.




$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_y 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_x & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$


