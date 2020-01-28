# Hibert-Schmidt Independence Criterion (HSIC)

---
## Resources

**Presentations**

* [HSIC, A Measure of Independence](http://www.cmap.polytechnique.fr/~zoltan.szabo/talks/invited_talk/Zoltan_Szabo_invited_talk_EPFL_LIONS_28_02_2018_slides.pdf) - Szabo (2018)
* [Measuring Independence with Kernels](https://harpo.uv.es/wiki/_media/dspg-ipl:gcamps_hsic.pdf) - Gustau


---
### Hilbert-Schmidt Norm (Frobenius Norm)

Let's have the two distributions $\mathcal{X} \in \mathbb{R}^{D_x}$ and $\mathcal{Y} \in \mathbb{R}^{D_y}$. Let's also assume that we can sample $(x,y)$ from $\mathbb{P}_{xy}$. We can capture the first order dependencies between $X$ and $Y$ by the covariance matrix which is defined as:

$$C_{xy} = \mathbb{E}_{xy}(xy^\top) - \mathbb{E}_x(x)\mathbb{E}_y(y^\top)$$

We can use the Hilbert-Schmidt Norm (HS-Norm) as a statistic to effectively summarize content within this covariance matrix. It's defined as:

$$||C_{xy}||_{\mathcal{H}}^2 = \sum_i \lambda_i^2 = \text{tr}\left[ C_{xy}^\top C_{xy} \right]$$

where $\lambda_i$ are the eigenvalues of $C_{xy}$. Note that this term is zero iff $X$ and $Y$ are independent and greater than zero otherwise. Since the covariance matrix is a first-order measure of the relations, we can only summarize the the first order relation information. 

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

This is very easy to compute in practice. One just needs to calculate the Frobenius Norm (Hilbert-Schmidt Norm) of a covariance matrix This boils down to computing the trace of the matrix multiplication of two matrices: $tr(C_{xy}^\top C_{xy})$. So in algorithmically that is:

```python
hsic_score = np.sqrt(np.trace(C_xy.T * C_xy))
```
We can make this faster by using the `sum` operation

```python
# Numpy
hsic_score = np.sqrt(np.sum(C_xy * C_xy))
# PyTorch
hsic_score = (C_xy * C_xy).sum().sum()
```

**Refactor**

There is a built-in function to be able to to speed up this calculation by a magnitude.

```python
hs_score = np.linalg.norm(C_xy, ord='fro')
```

and in PyTorch

```python
hs_score = torch.norm(C_xy, p='fro)
```

</details>

---

### Hilbert-Schmidt Criterion

Let's assume there exists a nonlinear mapping from our data space to the Hilbert space. So $\phi : \mathcal{X} \rightarrow \mathcal{F}$ and $\psi : \mathcal{Y} \rightarrow \mathcal{G}$. We also assume that there is a representation of this mapping via the dot product between the features of the data space; i.e. $K_x(x,x') = \langle \phi(x), \phi(x') \rangle$ and $K_y(y,y') = \langle \psi(y), \psi(y') \rangle$. So now the data matrices are $\Phi \in \mathbb{R}^{N\times N_\mathcal{F}}$ and $\Psi \in \mathbb{R}^{N \times N_\mathcal{G}}$. So we can take the kernelized version of the cross covariance mapping as defined for the covariance matrix:

$$
\begin{aligned}
||C_{\phi(x)\psi(x)}||_\mathcal{H}^2 
&= ||\Phi^\top \Psi||^2_{\mathcal{F}} 
\end{aligned}$$

Now after a bit of simplication, we end up with the HSIC-Norm:

$$
\begin{aligned}
\text{HSIC}(\hat{P}_{XY}, \mathcal{F}, \mathcal{G})
&= tr (K_{\mathbf{x}}K_{\mathbf{x}})
\end{aligned}$$



<details>

<summary>
    <font color="red">Proof
    </font>
</summary>

In this section, we will derive the empirical formula for HSIC using the Hilbert-Schmidt Norm of the covariance matrix with the kernel mapping.

$$
\begin{aligned}
||C_{\phi(x)\psi(x)}||_\mathcal{H}^2 
&= ||\Phi^\top \Psi||^2 \\
&= tr\left[ (\Phi^\top \Psi)^\top (\Phi^\top \Psi)\right] \\
&= tr \left[ \Psi^\top \Phi \Phi^\top \Psi\right] \\
&= tr \left[ \Psi \Psi^\top \Phi \Phi^\top \right] \\
&= tr (K_{\mathbf{x}}K_{\mathbf{x}})
\end{aligned}
$$

</details>


<details>

<summary>
    <font color="black">Details
    </font>
</summary>

Using the same argument as above, we can also define a cross covariance matrix of the form:

$$C_{xy} = \mathbb{E}_{xy} \left[  (\phi(x) - \mu_x) \otimes (\psi(y) - \mu_y)\right]$$

where $\otimes$ is the tensor product, $\mu_x, \mu_y$ are the expecations of the mappings $\mathbb{E}_x [\phi (x)]$, $\mathbb{E}_y[\psi(y)]$ respectively. The HSIC is the cross-covariance operator described above and can be expressed in terms of kernels.

$$\text{HSIC}(\mathcal{F}, \mathcal{G}, \mathbb{P}_{xy}) = ||C_{xy}||_{\mathcal{H}}^2$$
$$\text{HSIC}(\mathcal{F}, \mathcal{G}, \mathbb{P}_{xy}) = \mathbb{E}_{xx',yy'} \left[ K_x(x,x')K_y(y,y') \right] $$
$$+  \mathbb{E}_{xx'} \left[ K_x(x,x')\right] \mathbb{E}_{yy'} \left[ K_y(y,y')\right]$$
$$-  2\mathbb{E}_{xy} \left[ \mathbb{E}_{x'} \left[ K_x(x,x')\right] \mathbb{E}_{y'} \left[ K_y(y,y')\right] \right]$$

where $\mathbb{E}_{xx'yy'}$ is the expectation over both $(x,y) \sim \mathbb{P}_{xy}$ and we assume that $(x',y')$ can be sampled independently from $\mathbb{P}_{xy}$.

</details>


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

This is very easy to compute in practice. One just needs to calculate the Frobenius Norm (Hilbert-Schmidt Norm) between two kernel matrics that correctly model your data. This boils down to computing the trace of the matrix multiplication of two matrices: $tr(K_x^\top K_y)$. So in algorithmically that is

```python
hsic_score = np.trace(K_x.T @ K_y)
```

Notice that this is a 3-part operation. So, of course, we can refactor this to be much easier. A faster way to do this is:

```python
hsic_score = np.sum(K_x * K_y)
```

This can be orders of magnitude faster because it is a much cheaper operation to compute elementwise products than a sum. And for fun, we can even use the `einsum` notation.

```python
hsic_score = np.einsum("ji,ij->", K_x, K_y)
```

</details>


---
## Practical Equations


**HSIC**

$$
\text{MMD}^2(\hat{P}_{XY}, \hat{P}_X\hat{P}_Y, \mathcal{H}_k) \coloneqq 
\frac{1}{n^2}\text{tr} \left( K_x H K_y H \right)
$$

where $H$ is the centering matrix $H=I_n-\frac{1}{n}1_n1_n^\top$.

$$
\text{HSIC}^2(\hat{P}_{XY}, \mathcal{F}, \mathcal{G}) = 
\text{MMD}^2(\hat{P}_{XY}, \hat{P}_X\hat{P}_Y, \mathcal{H}_k)
$$

$$$$

---
### Objects of Interest


#### Mean Embedding

$$\mu_k(\mathbb{P}) \coloneqq \int_\mathcal{X} \underbrace{\psi(x)}_{k(\cdot,x)} d\mathbb{P}(x) \in \mathcal{H}_k$$

#### Maximum Mean Discrepency (MMD)

$$\text{MMD}_k(\mathbb{P}, \mathbb{Q}) \coloneqq || \mu_k(\mathbb{P}) - \mu_k(\mathbb{Q}) \in ||_{\mathcal{H}_k}$$


#### Hilbert-Schmidt Independence Criterion (HSIC)

$$\text{HSIC}_k(\mathbb{P}) \coloneqq \text{MMD}_k(\mathbb{P}, \otimes_{m=1}^M \mathbb{P}_m)$$


---
### Linear Algebra

**Norm** induced by the inner product: 

$$||f||_{\mathcal{H}} \coloneqq \sqrt{\langle f,f \rangle_{\mathcal{H}}}$$



---
### Classical Information Theory

**Kullback-Leibler Divergence**

$$D_{KL}(\mathbb{P}, \mathbb{Q}) = \int_{\mathbb{R}^d} p(x) \log \left[ \frac{p(x)}{q(x)} \right]dx$$

**Mutual Information**

$$I(\mathbb{P})=D_{KL}\left( \mathbb{P}, \otimes_{m=1}^{M}\mathbb{P}_m \right)$$


---
### Tangent Kernel Alignment

**HSIC**


$$A(K_x, K_y) = 
\left\langle H K_x, H K_y \right\rangle_{F}
$$

**Original Kernel Tangent Alignment**

$$A(K_x, K_y) = 
\frac{\left\langle K_x, K_y \right\rangle_{F}}{\sqrt{|| K_x||_{F}|| K_y ||_{F}}}
$$

The alignment can be seen as a similarity score based on the cosine of the angle. For arbitrary matrices, this score ranges between -1 and 1. But using positive semidefinite Gram matrices, the score is lower-bounded by 0.

**Centered Kernel Tangent Alignment**

$$A(H K_{x}, H K_{y}) = 
\frac{\left\langle H K_{x}, H K_{y} \right\rangle_{F}}{\sqrt{|| H K_{x}||_{F}|| H K_{y} ||_{F}}}
$$

They add a normalization term to deal with some of the shortcomings of the original KTA algorithm which had some benefits e.g. a way to cancel out unbalanced class effects. The improvement over the original algorithm seems minor but there is a critical difference. Without the centering, the alignment does not correlate well to the performance of the learning machine. 


---
### Covariance vs Correlation

The above methods can be put into perspective of the difference between covariance measures and correlation meausres. In this section, I will explain why that is so. We can write out the full definitions for covariance and correlation. The definition of covariance is:

$$\text{cov}_{XY}=\sigma_{XY}=E\left[(X - \mu_X)(Y - \mu_Y) \right]$$

The definition of correlation is:

$$\text{corr}_{XY}=\rho_{XY}=\frac{\sigma_{XY}}{\sigma_X \sigma_Y}$$

The correlation measure is dimensionless whereas the covariance is in units obtained by multiplying the units of the two variables.

**Source**: 

* [Wikipedia](https://en.wikipedia.org/wiki/Covariance_and_correlation) on Covariance and correlation
* [Wikipedia](https://en.wikipedia.org/wiki/Correlation_and_dependence) on Correlation and dependence


---
## Literature Review


**An Overview of Kernel Alignment and its Applications** - Wang et al (2012) - [PDF](https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf)

This goes over the literature of the kernel alignment method as well as some applications it has been used it.


### Applications

* Kerneel Target Alignment Parameter: A New Modelability for Regression Tasks - Marcou et al (2016) - [Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00539)
* Brain Activity Patterns - [Paper](https://www.frontiersin.org/articles/10.3389/fnins.2017.00550/full)
* Scaling - [Paper](https://link.springer.com/article/10.1007/s11222-016-9721-7)


### Textbooks

* Kernel Methods for Digital Processing - [Book](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118705810)

---

### Useful Formulas

**Kernel Alignment**

Empirical Alignment evaluates the similarity between the corresponding matrices.

$$
\begin{aligned}
A(K_1, K_2) &= \frac{\langle K_1, K_2 \rangle_{F}}{\sqrt{\langle K_1, K_1 \rangle_{F}\langle K_2, K_2 \rangle_{F}}}
\end{aligned}
$$

where:

$$\langle K_1, K_2 \rangle_{F} = \sum_{i=1}^N\sum_{j=1}^Nk_1(x_i, x_j)k_2(x_i, x_j)$$

This can be seen as a similarity score between the cosine of the angle. It has a lower bound of 0 because we typically only use positive semi-definite Gram matrices.

**Centered Kernel Alignment**

To counter the unbalanced class distribution:

$$
\begin{aligned}
k_c(x,z) = 
\left( \phi(x) - \mathbb{E} \left[\phi(X)\right] \right)^\top
\left( \phi(z) - \mathbb{E} \left[\phi(Z)\right] \right)
\end{aligned}
$$

The empirical centered alignment can be written as:

$$
\begin{aligned}
A_c(K_{c1}, K_{c2}) &= \frac{\langle K_{c1}, K_{c2} \rangle_{F}}{\sqrt{\langle K_{c1}, K_{c1} \rangle_{F}\langle K_{c2}, K_{c2} \rangle_{F}}}
\end{aligned}
$$

### Frobenius Norm (or Hilbert-Schmidt Norm) a matrix

$$
\begin{aligned}
||A|| &= \sqrt{\sum_{i,j}|a_{ij}|^2} \\
&= \sqrt{\text{tr}(A^\top A)} \\
&= \sqrt{\sum_{i=1}\lambda_i^2}
\end{aligned}$$


<!-- <details> -->
<summary>
    <font color="black">Details
    </font>
</summary>

Let $A=U\Sigma V^\top$ be the Singular Value Decomposition of A. Then

$$||A||_{F}^2 = ||\Sigma||_F^2 = \sum_{i=1}^r \lambda_i^2$$

If $\lambda_i^2$ are the eigenvalues of $AA^\top$ and $A^\top A$, then we can show 

$$
\begin{aligned}
||A||_F^2 &= tr(AA^\top) \\
&= tr(U\Lambda V^\top V\Lambda^\top U^\top) \\
&= tr(\Lambda \Lambda^\top U^\top U) \\
&= tr(\Lambda \Lambda^\top) \\
&= \sum_{i}\lambda_i^2
\end{aligned}
$$

<!-- </details> -->