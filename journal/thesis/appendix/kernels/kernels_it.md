# Kernels and Information Measures

This post will be based off of the paper from the following papers:

1. Measures of Entropy from Data Using Infinitely Divisible Kernels - Giraldo et. al. (2014)
2. Multivariate Extension of Matrix-based Renyi's $\alpha$-order Entropy Functional - Yu et. al. (2018)

## Short Overview

The following IT measures are possible with the scheme mentioned above:

1. Entropy
2. Joint Entropy
3. Conditional Entropy
4. Mutual Information


## Kernel Matrices

$$\begin{aligned}
\hat{f}(x) &= \frac{1}{N} \sum_{i=1}^N K_\sigma (x, x_i) \\
K_\sigma(x_i, x_j) &= \frac{1}{(2\pi \sigma^2)^{d/2}}\exp\left( - \frac{||x-x_i||^2_2}{2\sigma^2} \right)  
\end{aligned}$$

## Entropy


### $\alpha=1$

In this case, we can show that for kernel matrices, the Renyi entropy formulation becomes the eigenvalue decomposition of the kernel matrix.


$$
\begin{aligned}
H_1(x) &= \log \int_\mathcal{X}f^1(x) \cdot dx \\

\end{aligned}$$


### $\alpha=2$

In this case, we will have the Kernel Density Estimation procedure.

$$H_2(x) = \log \int_\mathcal{X}f^2(x)\cdot dx$$

$$H_2(x) = - \log \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^N K_{\sqrt{2}\sigma}(x_i, x_j)$$

$$H_2(x) = - \log \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^N K_{\sqrt{2}\sigma}$$

**Note**: We have to use the convolution theorem for Gaussian functions. [Source](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjq6ZGs3P3jAhVRasAKHZ24BCcQFjABegQIDBAF&url=http%3A%2F%2Fwww.tina-vision.net%2Fdocs%2Fmemos%2F2003-003.pdf&usg=AOvVaw1SaNhee0xBCB561s0D8Jba) | 

#### Practically

We can calculate this above formulation by simply multiplying the kernel matrix $K_x$ by the vector $1_N$.

$$\hat{H}_2(x) = - \log \frac{1}{N^2} \mathbf{1}_N^\top \mathbf{K}_x \mathbf{1}_N$$

where $\mathbf{1}_N \in \mathbf{R}^{1 \times N}$. The quantity $\mathbf{1}_N^\top \mathbf{K}_x \mathbf{1}_N$ is known as the *information potential*, $V$.

---
## Cross Information Potential RKHS

$$\hat{H}(X) = -log\left(\frac{1}{n^2} \text{tr}(KK)  \right) + C(\sigma)$$

* Cross Information Potential $\mathcal{V}$


---
## Joint Entropy

This formula uses the above entropy formulation. To incorporate both r.v.'s $X,Y$, we construct two kernel matrices $A,B$ respectively 

$$S_\alpha(A,B) = S_\alpha \left( \frac{A \circ B}{\text{tr}(A \circ B)} \right)$$

**Note**: 
* The trace is there for normalization.
* The matrices $A,B$ have to be the same size (due to the Hadamard product).

### Multivariate 

This extends to multiple variables. Let's say we have L variables, then we can calculate the joint entropy like so:

$$S_\alpha(A_1, A_2, \ldots, A_L) = S_\alpha \left( \frac{A_1 \circ A_2 \circ \ldots \circ A_L}{\text{tr}(A_1 \circ A_2 \circ \ldots \circ A_L)} \right)$$

---
## Conditional Entropy

This formula respects the traditional formula for conditional entropy; the joint entropy of r.v. $X,Y$ minus the entropy of r.v. $Y$, ($H(X|Y) = H(X,Y) - H(Y)$). Assume we have the kernel matrix for r.v. $X$ as $A$ and the kernel matrix for r.v. $Y$ as $B$. The following formula shows how this is calculated using kernel functions.

$$S_\alpha(A|B) = S_\alpha \left( \frac{A \circ B}{\text{tr}(A \circ B)} \right) - S_\alpha(B)$$

---
## Mutual Information

The classic Shannon definition is the sum of the marginal entropies mines the intersection between the r.v.'s $X,Y$, i.e. $MI(X;Y)=H(X)+H(Y)-H(X,Y)$. The following formula shows the MI with kernels:

$$I_\alpha(A;B) = S_\alpha(A) + S_\alpha(B) - S_\alpha \left( \frac{A \circ B}{\text{tr}(A \circ B)} \right)$$

The definition is the exactly the same and utilizes the entropy and joint entropy formulations above. 

### Multivariate

This can be extended to multiple variables. Let's use the same example for multi-variate solutions. Let's assume $B$ is univariate but $A$ is multivariate, i.e. $A \in \{A_1, A_2, \ldots, A_L \}$. We can write the MI as:

$$I_\alpha(A;B) = S_\alpha(A) + S_\alpha(B) - S_\alpha \left( \frac{A_1 \circ A_2 \circ \ldots \circ A_L \circ B}{\text{tr}(A_1 \circ A_2 \circ \ldots \circ A_L \circ B)} \right)$$


---
## Total Correlation

This is a measure of redundancy for multivariate data. It is basically the entropy of each of the marginals minus the joint entropy of the multivariate distribution. Let's assume we have $A$ as a multivarate distribution, i.e. $A \in \{A_1, A_2, \ldots, A_L \}$. Thus we can write this distribution using Kernel matrices:


$$T_\alpha(\mathbf{A}) = 
H_\alpha(A_1) + H_\alpha(A_2) + \ldots + H_\alpha(A_d) 
- H_\alpha \left( \frac{A_1 \circ A_2 \circ \ldots \circ A_d}{\text{tr}(A_1 \circ A_2 \circ \ldots \circ A_d)} \right)$$

