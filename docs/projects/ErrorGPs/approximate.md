---
title: Gaussian Approximations
description: All of my projects
authors:
    - J. Emmanuel Johnson
path: docs/projects/ErrorGPs
source: approximate.md
---
# Error Propagation in Gaussian Transformations

We're in the setting where we have some inputs that come from a distribution $\mathbf{x} \sim \mathcal{N}(\mathbf{\mu_x}, \Sigma_\mathbf{x})$ and we have some outputs $y\in\mathbb{R}$. Typically we have some function $f(\mathbf)$ that maps the points $f:\mathbb{R}^D \rightarrow \mathbb{R}$. So like we do with GPs, we have the following function"

$$
y=f(\mathbf{x})
$$

---

## Change of Variables

To have this full mapping, we would need to use the change of variables method because we are doing a transformation between of probability distributions.

$$
\begin{aligned}
p(y) &= p(x)\left| \frac{dy}{dx} \right|^{-1} \\
&= p(f^{-1}(y)|\mu_x,\Sigma_x)|\nabla f(y)|
\end{aligned}
$$

This is very expensive and can be very difficult depending upon the function.

---

## Conditional Gaussian Distribution

Alternatively, if we know that $f()$ is described by a Gaussian distribution, then we can find the joint distribution between $\mathbf{x},y$. This can be described as:

$$
\begin{bmatrix}
    \mathbf{x} \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_\mathbf{x} \\ 
    f(\mathbf{x})
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_\mathbf{x} & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$

---

## Taylor Expansions

 by using a Taylor series expansion. Let $\mu_x$ be the true inputs and we perturb these by some noise $\delta_x$ which is described by a normal distribution $\delta_x \sim \mathcal{N}(0, \Sigma_x)$. So we can write an expression for the function $f(\mathbf{x})$.

$$
\begin{aligned}
f(\mathbf{x}) &= f(\mu_x + \delta_x) \\
&\approx f(\mu_x) + \nabla_x f(\mu_x)\delta_x + \frac{1}{2}\sum_i \delta_x^\top \nabla_{xx}^{(i)}f(\mu_x)\delta_x e_i + \ldots
\end{aligned}
$$

where $\nabla_x f(\mu_x)$ is the jacobian of $f(\cdot)$ evaluated at $\mu_x$, $\nabla_{xx}f(\mu_x)$ is the hessian of $f(\cdot)$ evaluated at $\mu_x$ and $e_i$ is a ones vector (essentially the trace of a full matrix).

---

### Joint Distribution

So, we want $\tilde{f}(\mathbf{x})$ which is a joint distribution of $\begin{bmatrix} \mathbf{x} \\ f(\mathbf{x})\end{bmatrix}$. But as we said, this is difficult to compute so we do a Taylor approximation to this function $\begin{bmatrix} \mu_\mathbf{x} \\ f(\mu_\mathbf{x} + \delta_x) \end{bmatrix}$. But we still want the **full joint distribution**, ideally Gaussian. So that would mean we at least need the expectation and the covariance.

$$
\mathbb{E}_\mathbf{x}\left[ \tilde{f}(\mathbf{x}) \right], \mathbb{V}_\mathbf{x}\left[ \tilde{f}(\mathbf{x}) \right]
$$

??? details "Derivation"

    === "Mean Function"

        The mean function is the easiest to derive. We can just take the expectation of the first two terms and we'll see why all the higher order terms disappear.

        $$
        \begin{aligned}
        \mathbb{E}_\mathbf{x}\left[ \tilde{f}(\mathbf{x}) \right] &=
        \mathbb{E}_\mathbf{x}\left[ \tilde{f}(\mu_\mathbf{x})  + \nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x} \right] \\
        &= \mathbb{E}_\mathbf{x}\left[ \tilde{f}(\mu_\mathbf{x}) \right]  +
        \mathbb{E}_\mathbf{x}\left[ \nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x} \right] \\
        &= \tilde{f}(\mu_\mathbf{x})  +
        \nabla_\mathbf{x}\mathbb{E}_\mathbf{x}\left[ f(\mathbf{x})\epsilon_\mathbf{x} \right] \\
        &= \tilde{f}(\mu_\mathbf{x})\\
        \end{aligned}
        $$

    === "Covariance Function"

        The covariance function is a bit harder. But again, the final expression is quite simple.

        $$
        \begin{aligned}
        \mathbb{V}_\mathbf{x}\left[ \tilde{f}(\mathbf{x}) \right] &=
        \mathbb{E}_\mathbf{x}\left[f(\mathbf{x}) - \mathbb{E}_\mathbf{x}\left[f(\mathbf{x})\right] \right] \;
        \mathbb{E}_\mathbf{x}\left[f(\mathbf{x}) - \mathbb{E}_\mathbf{x}\left[f(\mathbf{x})\right] \right]^\top \\
        &\approx
        \mathbb{E}_\mathbf{x}
        \left[f(\mathbf{x}) - f(\mu_\mathbf{x}) \right] \;
        \mathbb{E}_\mathbf{x}
        \left[f(\mathbf{x}) - f(\mu_\mathbf{x}) \right]^\top\\
        &\approx
        \mathbb{E}_\mathbf{x}
        \left[f(\mu_\mathbf{x}) + \nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x} - f(\mu_\mathbf{x}) \right] \;
        \mathbb{E}_\mathbf{x}
        \left[f(\mu_\mathbf{x}) + \nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x}- f(\mu_\mathbf{x}) \right]^\top\\
        &\approx
        \mathbb{E}_\mathbf{x}
        \left[\nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x} \right] \;
        \mathbb{E}_\mathbf{x}
        \left[\nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x} \right]^\top\\
        &\approx
        \mathbb{E}_\mathbf{x}
        \left[\left(\nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x}\right) \left(\nabla_\mathbf{x}f(\mu_\mathbf{x})\epsilon_\mathbf{x}\right)^\top \right] \\
        &\approx
        \nabla_\mathbf{x}f(\mu_\mathbf{x})
        \mathbb{E}_\mathbf{x}
        \left[\left(\epsilon_\mathbf{x}\right) \left(\epsilon_\mathbf{x}\right)^\top \right]
        \nabla_\mathbf{x}f(\mu_\mathbf{x}) \\
        &\approx
        \nabla_\mathbf{x}f(\mu_\mathbf{x})
        \Sigma_\mathbf{x}
        \nabla_\mathbf{x}f(\mu_\mathbf{x})
        \end{aligned}
        $$

So now we can have a full expression for the joint distribution with the Taylor expansion.

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
    \Sigma_\mathbf{x} & C_\mathcal{T} \\
    C_\mathcal{T}^\top & \Pi_\mathcal{T}
    \end{bmatrix}
    \right)
$$

where:

$$
\begin{aligned}
\mu_y &= f(\mu_\mathbf{x})\\
C_T &= \Sigma_\mathbf{x} \nabla_\mathbf{x}f(\mu_\mathbf{x}) \\
\Pi_T &= 
\nabla_\mathbf{x}f(\mu_\mathbf{x})
\Sigma_\mathbf{x} 
\nabla_\mathbf{x}f(\mu_\mathbf{x}) \\
\end{aligned}
$$

---

## Joint Distribution for Additive Noise

So in this setting, we are closer to what we typically use for a GP model. 

$$
y = f(\mathbf{x}) + \epsilon
$$

where we have noisy inputs with additive Gaussian noise$\mathbf{x} \sim \mathcal{N}(\mathbf{\mu_x}, \Sigma_\mathbf{x})$ and we assume additive Gaussian noise for the outputs $\epsilon \sim \mathcal{N}(0, \sigma_y^2 \mathbf{I})$. We want a joint distribution of $\mathbf{x},y$. So using the same sequences of steps as above, we actually get a very similar joint distribution, just with an additional term.

$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_{\mathcal{GP}} 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_\mathbf{x} & C_\mathcal{eGP} \\
    C_\mathcal{eGP}^\top & \Pi_\mathcal{eGP}
    \end{bmatrix}
    \right)
$$

where

$$
\begin{aligned}
\mu_y &= f(\mu_x) \\
\Pi_\mathcal{eGP} &= \nabla_x f(\mu_x) \: \Sigma_x \: \nabla_x f(\mu_x)^\top + \nu^2(x) \\
C_\mathcal{eGP} &= \Sigma_x \: \nabla_x^\top f(\mu_x)
\end{aligned}
$$

So if we want to make predictions with our new model, we will have the final equation as:

$$
\begin{aligned}
f &\sim \mathcal{N}(f|\mu_{GP}, \nu^2_{GP}) \\
    \mu_\mathcal{eGP} &= K_{*} K_{GP}^{-1}y=K_{*} \alpha \\
    \sigma^2_\mathcal{eGP} &= K_{**} - K_{*} K_{GP}^{-1}K_{*}^{\top} + \tilde{\Sigma}_x
\end{aligned}
$$

where $\tilde{\Sigma}_x = \nabla_x \mu_{GP} \Sigma_x \nabla \mu_{GP}^\top$.
