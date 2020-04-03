# Maximum Mean Discrepancy (MMD)

> The Maximum Mean Discrepency (MMD) measurement is a distance measure between feature means.

---

- [Idea](#idea)
- [Formulation](#formulation)
  - [<font color="red">Proof</font>](#font-color%22red%22prooffont)
- [Kernel Trick](#kernel-trick)
- [Empirical Estimate](#empirical-estimate)
  - [<font color="blue">Code</font>](#font-color%22blue%22codefont)
- [Equivalence](#equivalence)
  - [Euclidean Distance](#euclidean-distance)
  - [KL-Divergence](#kl-divergence)
  - [Variation of Information](#variation-of-information)
  - [HSIC](#hsic)
  - [**<summary><font color="red">Proof</font></summary>**](#summaryfont-color%22red%22prooffontsummary)
- [Resources](#resources)

---

## Idea

This is done by taking the between dataset similarity of each of the datasets individually and then taking the cross-dataset similarity.

---

## Formulation

$$
\begin{aligned}
MMD^2(P,Q) 
&= ||\mu_P - \mu_Q||_\mathcal{F}^2 \\
&= \mathbb{E}_{\mathcal{X} \sim P}\left[ k(x,x')\right] +
\mathbb{E}_{\mathcal{Y} \sim Q}\left[ k(y,y')\right] -
2 \mathbb{E}_{\mathcal{X,Y} \sim P,Q}\left[ k(x,y)\right]
\end{aligned}
$$

### <font color="red">Proof</font>

$$
\begin{aligned}
||\mu_P - \mu_Q||_\mathcal{F}^2 
&= 
\langle \mu_P - \mu_Q, \mu_P - \mu_Q \rangle_\mathcal{F} \\
&= 
\langle \mu_P, \mu_P \rangle_\mathcal{F} +
\langle \mu_Q, \mu_Q \rangle_\mathcal{F} -
2 \langle \mu_P,\mu_Q \rangle_\mathcal{F} \\
&= 
\mathbb{E}_{\mathcal{X} \sim P} \left[ \mu_Q(x) \right] +
\mathbb{E}_{\mathcal{Y} \sim Q} \left[ \mu_P(y) \right] -
2 \mathbb{E}_{\mathcal{X} \sim P, Y \sim Q} \left[ \mu_P(x) \right] ??? \\
&= 
\mathbb{E}_{\mathcal{X} \sim P} \langle \mu_P, \varphi(x) \rangle_\mathcal{F} +
\mathbb{E}_{\mathcal{Y} \sim Q} \langle \mu_Q, \varphi(y) \rangle_\mathcal{F} -
2 ... ??? \\
&= 
\mathbb{E}_{\mathcal{X} \sim P} \langle \mu_P, k(x, \cdot) \rangle_\mathcal{F} +
\mathbb{E}_{\mathcal{Y} \sim Q} \langle \mu_Q, k(y, \cdot) \rangle_\mathcal{F} -
2 ... ??? \\
&= 
\mathbb{E}_{\mathcal{X} \sim P} \left[ k(x,x') \right] +
\mathbb{E}_{\mathcal{Y} \sim Q} \left[ k(y,y') \right] -
2 \mathbb{E}_{\mathcal{X,Y} \sim P,Q } \left[ k(x,y) \right]
\end{aligned}
$$



---

## Kernel Trick

Let $k(X,Y) = \langle \varphi(x), \varphi(y) \rangle_\mathcal{H}$:

$$
\begin{aligned}
\text{MMD}^2(P, Q) 
&=
|| \mathbb{E}_{X \sim P} \varphi(X) - \mathbb{E}_{Y \sim P} \varphi(Y) ||^2_\mathcal{H} \\
&=
\langle \mathbb{E}_{X \sim P} \varphi(X), \mathbb{E}_{X' \sim P} \varphi(X')\rangle_\mathcal{H} +
\langle \mathbb{E}_{Y \sim Q} \varphi(Y), \mathbb{E}_{Y' \sim Q} \varphi(Y')\rangle_\mathcal{H} -
2 \langle \mathbb{E}_{X \sim P} \varphi(X), \mathbb{E}_{Y' \sim Q} \varphi(Y')\rangle_\mathcal{H}
\end{aligned}
$$

**Source**: [Stackoverflow](https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution)

---

## Empirical Estimate

$$
\begin{aligned}
\widehat{\text{MMD}}^2 &= 
\frac{1}{n(n-1)} \sum_{i\neq j}^N k(x_i, x_j) + 
\frac{1}{n(n-1)} \sum_{i\neq j}^N k(y_i, y_j) -
\frac{2}{n^2} \sum_{i,j}^N k(x_i, y_j)

\end{aligned}
$$

### <font color="blue">Code</font>

```python
# Term 1
c1 = 1 / ( m * (m - 1))
A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

# Term II
c2 = 1 / (n * (n - 1))
B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

# Term III
c3 = 1 / (m * n)
C = np.sum(Kxy)

# estimate MMD
mmd_est = c1 * A + c2 * B - 2 * c3 * C
```

**Sources**

* [Douglas Sutherland](https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py)
* [HSIC BottleNeck](https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/math/hsic.py)
* [Eugene Belilovsky](https://github.com/eugenium/MMD/blob/master/mmd.py)

---

## Equivalence

### Euclidean Distance

Let's assume that $\mathbf{x,y}$ come from two distributions, so$\mathbf{x} \sim \mathbb{P}$ and $\mathbf{x} \sim \mathbb{Q}$. We can write the MMD as norm of the difference between the means in feature spaces.

$$
\text{D}_{ED}(\mathbb{P,Q}) 
= ||\mu_\mathbf{x} - \mu_\mathbf{y}||^2_F
= ||\mu_\mathbf{x}||^2_F + ||\mu_\mathbf{y}||^2_F -
2 \langle \mu_\mathbf{x}, \mu_\mathbf{y}\rangle_F
$$



**Empirical Estimation**

This is only good for Gaussian kernels. But we can empirically estimate this as:

$$\text{D}_{ED}(\mathbb{P,Q}) 
= \frac{1}{N_x^2} 
\sum_{i=1}^{N_x}\sum_{j=1}^{N_x} 
\text{G}(\mathbf{x}_i, \mathbf{x}_j) +
\frac{1}{N_y^2}
\sum_{i=1}^{N_y}\sum_{j=1}^{N_y}
\text{G}(\mathbf{y}_i, \mathbf{y}_j) -
2 \frac{1}{N_x N_y}
\sum_{i=1}^{N_x}\sum_{j=1}^{N_y}
\text{G}(\mathbf{x}_i, \mathbf{y}_j)
$$

where G is the Gaussian kernel with a standard deviation of $\sigma$. 

* Information Theoretic Learning: Renyi's Entropy and Kernel Perspectives - Principe

---

### KL-Divergence

This has some alternative interpretation that is similar to the Kullback-Leibler Divergence. Remember, the MMD is the distance between the joint distribution $P=\mathbb{P}_{x,y}$ and the product of the marginals $Q=\mathbb{P}_x\mathbb{P}_y$. 

$$\text{MMD}(P_{XY},P_X P_Y, \mathcal{H}_k) = || \mu_{PQ} - \mu_{P}\mu_{Q}||$$

This is similar to the KLD which has a similar interpretation in terms of the Mutual information: the difference between the joint distribution $P(x,y)$ and the product of the marginal distributions $p_x p_y$.

$$I(X,Y) = D_{KL} \left[ P(x,y) || p_x p_y \right]$$

---

### Variation of Information

In informaiton theory, we have a measure of variation of information (aka the shared information distance) which a simple linear expression involving mutual information. However, it is a valid distance metric that obeys the triangle inequality.

$$\text{VI}(X,Y) = H(X) + H(Y) - 2 I (X,Y)$$

where $H(X)$ is the entropy of $\mathcal{X}$ and $I(X,Y)$ is the mutual information between $\mathcal{X,Y}$.

**Properties**

* $\text{VI}(X,Y) \geq 0$
* $\text{VI}(X,Y) = 0 \implies X=Y$
* $\text{VI}(X,Y) = d(Y,X)$
* $\text{VI}(X,Z) \leq d(X,Y) + d(Y,Z)$

---

### HSIC

Similar to the KLD interpretation, this formulation is equivalent to the Hilbert-Schmidt Independence Criterion. If we think of the MMD distance between the joint distribution & the product of the marginals then we get the HSIC measure.


$$
\begin{aligned}
 \text{MMD}^2(P_{XY}, P_XP_Y; \mathcal{H}_k) &= ||\mu_{\mathbb{P}_{XY}} - \mu_{P_XP_Y}||
 \end{aligned}
$$


which is the exact formulation for HSIC.

$$
\begin{aligned}
 \text{MMD}^2(P_{XY}, P_XP_Y; \mathcal{H}_k) &=  \text{HSIC}^2(P_{XY}; \mathcal{F}, \mathcal{G})
\end{aligned}
$$

where we have some equivalences.




### **<summary><font color="red">Proof</font></summary>**

First we need to do some equivalences. First the norm of two feature spaces $\varphi(\cdot, \cdot)$ is the same as the kernel of the cross product.

$$
\begin{aligned}
\langle \varphi(x,y), \varphi(x,y) \rangle_\mathcal{F} &= k \left((x,y),(x',y')\right)
\end{aligned}
$$

The second is the equivalence of the kernel of the cross-product of $\mathcal{X,Y}$ is equal to the multiplication of the respective kernels for $\mathcal{X,Y}$. So, let's say we have a kernel $k$ on dataset $\mathcal{X}$ in the feature space $\mathcal{F}$. We also have a kernel $k$ on dataset $\mathcal{Y}$ with feature space $\mathcal{G}$. The kernel $k$ on the $\mathcal{X,Y}$ pairs are similar.

$$
\begin{aligned}
k\left((x,y),(x',y')\right) &= k(x,x')\,k(y,y') \\
\end{aligned}
$$




---

## Resources

* Arthur Grettons Lectures - [lec 5](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture5_distribEmbed_1.pdf) | [lec 2]()
  * [Notes on Mean Embedding](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture5_covarianceOperator.pdf)
  * [What is RKHS](http://www.stats.ox.ac.uk/~sejdinov/teaching/atml14/Theory_2014.pdf)
  * [The Maximum Mean Discrepancy for Training Generative Adversarial Networks](http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2019mva/gretton.pdf)
  * [From Zero to RKHS](http://users.umiacs.umd.edu/~hal/docs/daume04rkhs.pdf)
  * [RKHS in ML](http://users.umiacs.umd.edu/~hal/docs/daume04rkhs.pdf)
* Similarity Loss (TF2) - [code](https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/losses.py#L40)
* [MMD Smola](http://alex.smola.org/teaching/iconip2006/iconip_3.pdf)