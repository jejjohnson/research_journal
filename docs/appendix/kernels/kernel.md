# Kernel Measures of Similarity


**Notation**

* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{x}}$ are samples from a multidimentionsal r.v. $\mathcal{X}$
* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{y}}$ are samples from a multidimensional r.v. $\mathcal{Y}$
* $K \in \mathbb{R}^{N \times N}$ is a kernel matrix.
  * $K_\mathbf{x}$ is a kernel matrix for the r.v. $\mathcal{X}$
  * $K_\mathbf{y}$ is a kernel matrix for the r.v. $\mathcal{Y}$
  * $K_\mathbf{xy}$ is the cross kernel matrix for the r.v. $\mathcal{X,Y}$
* $\tilde{K} \in \mathbb{R}^{N \times N}$ is the centered kernel matrix.

**Observations**

* $\mathbf{X},\mathbf{Y}$ can have different number of dimensions
* $\mathbf{X},\mathbf{Y}$ must have different number of samples



### Feature Map

We have a function $\varphi(X)$ to map $\mathcal{X}$ to some feature space $\mathcal{F}$.

$$\phi(X) = \left[ \cdots, \varphi_i(x), \cdots \right] \in N$$

### Function Class

Reproducing Kernel Hilbert Space $\mathcal{H}$ with kernel $k$.

Evaluation functionals 

$$f(x) = \langle k(x,\cdot), f \rangle$$

We can compute means via linearity

$$
\begin{aligned}
\mathbb{E}_{X \sim P} \left[ f(X) \right] 
&=
\mathbb{E}_{X \sim P} \left[ \langle k(x, \cdot), f \rangle  \right] \\
&=
\bigg\langle \mathbb{E}_{X \sim P} \left[   k(x, \cdot)\right], f   \bigg\rangle \\
&=
\langle \mu_P, f \rangle
\end{aligned}
$$

And empirically

$$
\begin{aligned}
\frac{1}{N} \sum_{i=1}^N f(x_i)
&=
\frac{1}{N} \sum_{i=1}^N \langle k(x, \cdot), f \rangle \\
&=
\bigg\langle \frac{1}{N} \sum_{i=1}^N k(x, \cdot), f   \bigg\rangle \\
&=
\langle \mu_X, f \rangle
\end{aligned} 
$$

### Kernels

This allows us to not have to explicitly calculate $\varphi(X)$. We just need an algorithm that calculates the dot product between them.

$$\langle \varphi(X), \varphi(X') \rangle_\mathcal{F} = k(X, X')$$

### Reproducing Kernel Hilbert Space Notation

**Reproducing Property**

$$\langle f, k(x,\cdot) \rangle = f(x)$$

Equivalence between $\phi(x)$ and $k(x,\cdot)$.

$$\langle k(x, \cdot), k(x', \cdot) \rangle = k(x, x')$$

---

## Probabilities in Feature Space: The Mean Trick


#### Mean Embedding

$$\mu_k(\mathbb{P}) \coloneqq \int_\mathcal{X} \underbrace{\psi(x)}_{k(\cdot,x)} d\mathbb{P}(x) \in \mathcal{H}_k$$

#### Maximum Mean Discrepency (MMD)

$$\text{MMD}_k(\mathbb{P}, \mathbb{Q}) \coloneqq || \mu_k(\mathbb{P}) - \mu_k(\mathbb{Q}) \in ||_{\mathcal{H}_k}$$


#### Hilbert-Schmidt Independence Criterion (HSIC)

$$\text{HSIC}_k(\mathbb{P}) \coloneqq \text{MMD}_k(\mathbb{P}, \otimes_{m=1}^M \mathbb{P}_m)$$


---

Given $\mathbb{P}$ a Borel probability measure on $\mathcal{X}$, we can define a feature map $\mu_P \in \mathcal{F}$.

$$\mu_P = \left[ \ldots \mathbb{E}_P\left[ \varphi_i(\mathbf{x}) \right] \right]$$

Given a positive definite kernel $k(x,x')$, we can define the expectation of the cross kernel as:

$$\mathbb{E}_{P,Q}k(\mathbf{x,y}) = \langle \mu_P, \mu_Q \rangle_\mathcal{F}$$

for $x \sim P$ and $q \sim Q$. We can use the mean trick to define the following:

$$\mathbb{E}_P (f(X)) = \langle \mu_P, f(\cdot) \rangle_\mathcal{F}$$



---

## Covariance Measures

### Uncentered Kernel

 $$\text{cov}(\mathbf{X}, \mathbf{Y}) =||K_{\mathbf{xy}}||_\mathcal{F}
=\langle K_\mathbf{x}, K_\mathbf{y} \rangle_\mathcal{F}$$

---

### Centered Kernel

---

#### Hilbert-Schmidt Independence Criterion (HSIC)

$$\text{cov}(\mathbf{X}, \mathbf{Y}) =||\tilde{K}_{\mathbf{xy}}||_\mathcal{F}
=\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}$$

---

#### Maximum Mean Discrepency (MMD)


$$\text{cov}(\mathbf{X}, \mathbf{Y}) = ||K_\mathbf{x}||_\mathcal{F} + ||K_\mathbf{y}||_\mathcal{F}  -  2\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}$$

**[Source](https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/math/hsic.py#L69)**

---

## Kernel Matrix Inversion

#### Sherman-Morrison-Woodbury

$$(A + BCD)^{-1} = A^{-1} - A^{-1}B(C^{-1} + DA^{-1}B)^{-1}A^{-1}$$

**Matrix Sketch**

$$(LL^\top + \sigma I_N)^{-1} = \sigma^{-1} I_N - \sigma^{-1} (\sigma I_{n} + L^\top L)^{-1} L^{\top} $$

---

## Kernel Approximation

### Random Fourier Features

$$K \approx ZZ^\top$$


### Nystrom Approximation


$$K \approx C W^\dagger C^\top$$

According to ... the Nystroem approximation works better when you want features that are data dependent. The RFF method assumes a basis function and it is irrelevant to the data. It's merely projecting the data into the independent basis. The Nystroem approximation forms the basis through the data itself.

**Resources**

* A Practical Guide to Randomized Matrix Computations with MATLAB Implementations - Shusen Wang (2015) - [axriv](https://arxiv.org/abs/1505.07570)

### Structured Kernel Interpolation


$$
\begin{aligned}
K &\approx C W^\dagger C^\top \\
&\approx (XW) W^\dagger (XW)^\top \\
&\approx X W X^\top
\end{aligned}$$



---

## Correlation Measures

---

### Uncentered Kernel

---

#### Kernel Alignment (KA)

$$\rho(\mathbf{X}, \mathbf{Y})
=\frac{\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$

**In the Literature**

* Kernel Alignment

---

### Uncentered Kernel

---

#### Centered Kernel Alignment (cKA)

 $$\rho(\mathbf{X}, \mathbf{Y})
=\frac{\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$


**In the Literature**

* Centered Kernel Alignment


---

## Supplementary


---

## Ideas

**What happens when?**

* HS Norm of Noisy Matrix
* HS Norm of PCA components
