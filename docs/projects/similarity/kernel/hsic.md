# Hilbert-Schmidt Independence Criterion


* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Last Updated: 29 Jan, 2020

---

- [Resources](#resources)
- [Motivation](#motivation)
- [Covariance](#covariance)
  - [Gram Matrix](#gram-matrix)
- [Summarizing Information](#summarizing-information)
  - [Kernel Trick](#kernel-trick)
    - [Centering](#centering)
  - [Hilbert-Schmidt Independence Criterion](#hilbert-schmidt-independence-criterion)
- [Correlation](#correlation)
  - [Kernel Alignment](#kernel-alignment)
- [Connections](#connections)
  - [Mutual Information](#mutual-information)
- [Randomized HSIC](#randomized-hsic)
- [Practical Equations](#practical-equations)
  - [Objects of Interest](#objects-of-interest)
  - [Linear Algebra](#linear-algebra)
  - [Classical Information Theory](#classical-information-theory)
  - [Tangent Kernel Alignment](#tangent-kernel-alignment)
- [Future Outlook](#future-outlook)
- [Literature Review](#literature-review)
  - [Applications](#applications)
  - [Textbooks](#textbooks)
  - [Useful Formulas](#useful-formulas)

---

We use the Hilbert-Schmidt Independence Criterion (HSIC) measure independence between two distributions. It involves constructing an appropriate kernel matrix for each dataset and then using the Frobenius Norm as a way to "summarize" the variability of the data. Often times the motivation for this method is lost in the notorious paper of Arthur Gretton (the creator of the method), but actually, this idea was developed long before him with ideas from a covariance matrix perspective. Below are my notes for how to get from a simple covariance matrix to the HSIC method and similar ones.

---
## Resources

**Presentations**

* [HSIC, A Measure of Independence](http://www.cmap.polytechnique.fr/~zoltan.szabo/talks/invited_talk/Zoltan_Szabo_invited_talk_EPFL_LIONS_28_02_2018_slides.pdf) - Szabo (2018)
* [Measuring Independence with Kernels](https://harpo.uv.es/wiki/_media/dspg-ipl:gcamps_hsic.pdf) - Gustau

---

## Motivation

A very common mechanism to measure the differences between datasets is to measure the variability. The easiest way is the measure the covariance between the two datasets. However, this is limited to datasets with linear relationships and with not many outliers. Anscombe's classic dataset is an example where we have datasets with the same mean and standard deviation. 

<!-- <p align="center">
  <img src=".pics/demo_caseI_reg.png" alt="drawing" width="175"/>
  <img src="./pics/demo_caseII_reg.png" alt="drawing" width="175"/>
  <img src="./pics/demo_caseIII_reg.png" alt="drawing" width="175"/>
  <img src="./pics/demo_caseIV_reg.png" alt="drawing" width="175"/>
</p> -->

<p align="center">
  <img src="kernel/pics/demo_caseI_reg.png" alt="drawing" width="175"/>
  <img src="kernel/pics/demo_caseII_reg.png" alt="drawing" width="175"/>
  <img src="kernel/pics/demo_caseIII_reg.png" alt="drawing" width="175"/>
  <img src="kernel/pics/demo_caseIV_reg.png" alt="drawing" width="175"/>
</p>

This means measures like the covariance and correlation become useless because they will yield the same result. This requires us to have more robust methods or to do some really good preprocessing to make models easier.

* Multivariate
* Higher-Order Relations
* Non-Linear Regresentation

## Covariance

The first measure we need to consider is the covariance. This can be used for a single variable $X$, the covariance, or the cross-covariance between multiple variables $X,Y$. Some key properties include:

* It is a measure of the joint variability between the datasets
* It is difficult to interpret because it can range from $-\infty$ to $\infty$. 
* The units are dependent upon the inputs. 
* It is affected by isotropic scaling

For a univariate dataset we can use the following expression.

$$
\Sigma_\mathbf{xy} =
\mathbb{E}\left[(\mathbf{x}-\mu_\mathbf{x})(\mathbf{y}-\mu_\mathbf{y}) \right]
$$

We basically need to subtract the mean between all of the samples and then take the joint expectation of the two datasets. This will give us a covariance matrix of size $\Sigma_\mathbf{xy} \in \mathbb{R}^{D \times D}$. So this is a scalar value if $D=1$ and a matrix if $D>1$.


### Gram Matrix

We could also calculate the covariance between the sets of features, i.e. the Gram matrix.






## Summarizing Information

In the case of multidimensional datasets, this calculation because a bit more complicated. We don't get a scalar output; we get a covariance matrix $D\times D$. **Note**: this means that the matrix is in the feature space. The diagonal elements will be the covariance between each feature $d$ with itself and the off-diagonal elements will be the cross-covariance between each feature $d_i$ and another feature $d_j$. We can make some observations about the covariance matrices that can give us more information about the structure of our data. There are two important ones:

* A completely diagonal matrix means that all features are uncorrelated (orthogonal to each other)
* Diagonal Covariances are useful to learn (condition your model on) because it means that we have non-redundant features.



**Empirical Covariance**

We can empirically calculate this by doing the matrix multiplication of two vectors.



<details >
<summary>
    <font color="blue">Code</font>
</summary>

```python
C_xy = X.T @ X
```

</details>



**Observations**
* A completely diagonal covariance matrix means that all features are uncorrelated (orthogonal to each other).
* Diagonal covariances are useful for learning, they mean non-redundant features!


**Empirical Cross-Covariance**

This is the covariance between different datasets

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ Y
```

</details>

**Linear Kernel**

This measures the covariance between samples.

$$K_{xx} = X X^\top$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
K_xy = X @ X.T
```
</details>


---

### Kernel Trick

So now, we have only had a linear dot-similarity in the sample space of $\mathcal{X}$ and $\mathcal{Y}$. This is good but we can easily extend this to a non-linear transformation where we add an additional function $\psi$ for each of the kernel functions.

$$\langle C_{\mathbf{xx^\top }}, C_{\mathbf{yy^\top}}\rangle_{\mathcal{F}} 
=
\langle K_{\mathbf{x}}, K_{\mathbf{y}}\rangle_\mathcal{F}
$$

---

#### Centering



A very important but subtle point is that the method with kernels assumes that your data is centered in the kernel space. This isn't necessarily true. Fortunately it is easy to do so.

$$HK_xH = \tilde{K}_x$$

where $H$ is your centering matrix.

> Normalizing your inputs does **not** equal centering your kernel matrix.

<details>

We assume that the kernel function $\psi(x_i)$ has a zero mean like so:

$$\psi(x_i) = \psi(x_i) - \frac{1}{N}\sum_{r=1}^N \psi(x_r)$$

This holds if the covariance matrix is computed from $\psi(x_i)$. So the kernel matrix $K_{ij}=\psi(x_i)^\top \psi(x_j)$ needs to be replaced with $\tilde{K}_{ij}=\psi(x_i)^\top \psi(x_s)$ where $\tilde{K}_{ij}$ is:

$$
\begin{aligned}
\tilde{K}_{ij} 
&= \psi(x_i)^\top \psi(x_j) - 
\frac{1}{N} \sum_{r=1}^N 
- \frac{1}{N} \sum_{r=1}^N \psi(x_r)^\top \psi(x_j) 
+ \frac{1}{N^2} \sum_{r,s=1}^N \psi(x_r))^\top \psi(x_s) \\
&= K_{ij} - \frac{1}{N}\sum_{r=1}^{N}K_{ir} 
- \frac{1}{N} K_{rj}
+ \frac{1}{N^2} \sum_{r,s=1}^N K_s
\end{aligned}
$$


</details>

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

On a more practical note, this can be done easily by:

$$H = \mathbf{I}_N - \frac{1}{N} \mathbf{1}_N\mathbf{1}_N^\top$$

```python
H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples, n_samples)
```

**Refactor**

There is also a function in the `scikit-learn` library which does it for you.

```python
from sklearn.preprocessing import KernelCenterer

K_centered = KernelCenterer().fit_transform(K)
```

</details>

And like the covariance, we can also summarize the data structures with a correlation-like coefficient

$$\rho_\mathbf{xy}=\frac{ \langle K_{\mathbf{x}}, K_{\mathbf{y}}\rangle_\mathcal{F}}{||K_\mathbf{x}||_{\mathcal{F}} ||K_\mathbf{y}||_{\mathcal{F}}}$$


---




### Hilbert-Schmidt Independence Criterion

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

## Correlation

We can use this measure to summarize and normalize our data.

* It is a measure of relationship
* It is interpretable because it ranges from $-1,1$
* It isn't affected by isotropic scaling due to the normalization
* It is unitless because of the normalization

There are many benefits to using this measure which is why it is often used so widely.

$$\rho_{xy} = \frac{C_{xy}}{\sigma_x \sigma_y}$$

And the multidimensional case is:

$$
\begin{aligned}
\rho V(\mathbf{x,y}) 
&=
\frac{\langle \mathbf{W_x, W_y}\rangle_F}{||\mathbf{W_x}||_F \; ||\mathbf{W_y}||_F} \\
\end{aligned}
$$

### Kernel Alignment

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

## Connections


### Mutual Information

This is an approximation to kernel mutual information

$$I(X,Y) = - \frac{1}{2} \log \left( \frac{|C|}{|C_xx||C_yy||} \right)$$

$$I(X,Y) = - \frac{1}{2} \log (1- \rho^2)$$



---

## Randomized HSIC

If we understand that we can decompose a kernel matrix $K_x$ into two components, we can express it like so:

$$\tilde{K}_x = Z_x Z_x^\top$$

Let $\tilde{Z} = Z H$ be the normalized version. So plugging this decomposition into the HSIC formulation, we get:

$$
\text{HSIC}(\hat{P}_{XY}, \mathcal{F}, \mathcal{G}) \coloneqq 
\frac{1}{n^2}\text{tr} \left( \tilde{Z}_x \tilde{Z}_x^\top \tilde{Z}_y \tilde{Z}_y^\top \right)
$$

Of course this is still the same size as the original formulation so we haven't done anything effectively. But we can make use of the matrix property that the trace is invariant under cyclic permutations i.e.

$$\text{tr}(ABCD) = \text{tr} (BCDA) $$

So we can apply this cyclic transformation to obtain:

$$
\text{HSIC}(\hat{P}_{XY}, \mathcal{F}, \mathcal{G}) \coloneqq 
\frac{1}{n^2}\text{tr} \left(  \tilde{Z}_x^\top \tilde{Z}_y Z_y^\top \tilde{Z}_x \right)
$$

Now this final matrix is size $N_{\text{f}} \times N_{\text{f}}$ which can be smaller than the number of samples present $N_{\text{f}} << N$ which will result in an order $\mathcal{O}(N N_\text{f}^2)$.



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

---

## Future Outlook

**Advantages**

* Sample Space - Nice for High Dimensional Problems w/ a low number of samples
* HSIC can estimate dependence between variables of different dimensions
* Very flexible: lots of ways to create kernel matices

**Disadvantages**

* Computationally demanding for large scale problems
* Non-iid samples, e.g. speech or images
* Tuning Kernel parameters
* Why the HS nrm?


---
## Literature Review


**An Overview of Kernel Alignment and its Applications** - Wang et al (2012) - [PDF](https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf)

This goes over the literature of the kernel alignment method as well as some applications it has been used it.


### Applications

* Kerneel Target Alignment Parameter: A New Modelability for Regression Tasks - Marcou et al (2016) - [Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00539)
* Brain Activity Patterns - [Paper](https://www.frontiersin.org/articles/10.3389/fnins.2017.00550/full)
* Scaling - [Paper](https://link.springer.com/article/10.1007/s11222-016-9721-7)
* [Correlations Between Word Vector Sets](https://arxiv.org/pdf/1910.02902.pdf) - Zhelezniak et. al. (2019)

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

