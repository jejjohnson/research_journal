---
title: Overview
description: Overview of my similarity appendices
authors:
    - J. Emmanuel Johnson
path: docs/appendices/similarity
source: README.md
---
# Literature



## Paper I

**Linear Operators and Stochastic Partial Differential Equations in GPR** - Simo Särkkä - [PDF](https://users.aalto.fi/~ssarkka/pub/spde.pdf)

> Expresses derivatives of GPs as operators

[**Demo Colab Notebook**](https://colab.research.google.com/drive/1pbb0qlypJCqPTN_cu2GEkkKLNXCYO9F2)

He looks at ths special case where we have a GP with a mean function zero and a covariance matrix $K$ defined as:
$$
\mathbb{E}[f(\mathbf{x})f^\top(\mathbf{x'})] = K_{ff}(\mathbf{x,x'})
$$
So in GP terminology:
$$
f(\mathbf(x)) \sim \mathcal{GP}(\mathbf{0}, K_{ff}(\mathbf{x,x'}))
$$
We use the rulse for linear transformations of GPs to obtain the different transformations of the kernel matrix. 

Let's define the notation for the derivative of a kernel matrix. Let $g(\cdot)$ be the derivative operator on a function $f(\cdot)$. So:
$$
g(\mathbf{x}) = \mathcal{L}_x f(\mathbf{x})
$$

So now, we want to define the cross operators between the derivative $g(\cdot)$ and the function $f(\cdot)$. 

**Example**: He draws a distinction between the two operators with an example of how this works in practice. So let's take the linear operator $\mathcal{L}_{x}=(1, \frac{\partial}{\partial x})$. This operator:

* acts on a scalar GP $f(x)$
* a scalar input $x$ 
* a covariance function $k_{ff}(x,x')$ 
* outputs a scalar value $y$



We can get the following transformations:
$$
\begin{aligned}
K_{gf}(\mathbf{x,x'})
&= \mathcal{L}_x f(\mathbf{x}) f(\mathbf{x}) = \mathcal{L}_xK_{ff}(\mathbf{x,x'}) \\
K_{fg}(\mathbf{x,x'})
&= f(\mathbf{x}) f(\mathbf{x'}) \mathcal{L}_{x'} = K_{ff}(\mathbf{x,x'})\mathcal{L}_{x'} \\
K_{gg}(\mathbf{x,x'})
&= \mathcal{L}_x f(\mathbf{x}) f(\mathbf{x'}) \mathcal{L}_{x'}
= \mathcal{L}_xK_{ff}(\mathbf{x,x'})\mathcal{L}_{x'}^\top \\
\end{aligned}
$$

**Example**: The Cross-Covariance term $K_{fg}(\mathbf{x,x'})$

We can calculate the cross-covariance term $K_{fg}(\mathbf{x,x})$. We apply the following operation

$$
K_{fg}(x,x') = k_{ff}(\mathbf{x,x'})(1, \frac{\partial}{\partial x'})
$$
If we multiply the terms across, we get:
$$
K_{fg}(x,x') = k_{ff}(\mathbf{x,x'})\frac{\partial k_{ff}(\mathbf{x,x'})}{\partial x'}
$$
