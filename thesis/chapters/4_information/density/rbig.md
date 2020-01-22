# Rotation-Based Iterative Gaussianization (RBIG)


---

## Motivation

The RBIG algorithm is a member of the density destructor family of methods. A density destructor is a generative model that seeks to transform your original data distribution, $\mathcal{X}$ to a base distribution, $\mathcal{Z}$ through an invertible tranformation $\mathcal{G}_\theta$, parameterized by $\theta$.

$$\begin{aligned}
x &\sim \mathcal P_x \sim \text{Data Distribution}\\
\hat z &= \mathcal{G}_\theta(x) \sim \text{Approximate Base Distribution}
\end{aligned}$$

Because we have invertible transforms, we can use the change of variables formula to get probability estimates of our original data space, $\mathcal{X}$ using our base distribution $\mathcal{Z}$. This is a well known formula written as:

$$p_x(x)=
p_{z}\left( \mathcal{G}_{\theta}(x) \right)
\left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right| =
 p_{z}\left( \mathcal{G}_{\theta}(x) \right) \left| \nabla_x \mathcal{G}(x) \right|
$$

If you are familiar with normalizing flows, you'll find some similarities between the formulations. Inherently, they are the same. However most (if not all) of the major methods of normalizing flows, they focus on the log-likelihood estimation of data $\mathcal{X}$. They seek to minimize this log-deteriminant of the Jacobian as a cost function. RBIG is different in this regard as it has a different objective. RBIG seeks to maximize the negentropy or minimize the total correlation.

Essentialy, RVIG is an algorithm that respects the name density destructor fundamental. We argue that by destroying the density, we maximize the entropy and destroy all redundancies within the marginals of the variables in question. From this formulation, this allows us to utilize RBIG to calculate many other IT measures which we highlight below.


$$\begin{aligned}
z &\sim \mathcal P_z \sim \text{Base Distribution}\\
\hat x &= \mathbf G_\phi(x) \sim \text{Approximate Data Distribution}
\end{aligned}$$

---
## Algorithm

> Gaussianization - Given a random variance $\mathbf x \in \mathbb R^d$, a Gaussianization transform is an invertible and differentiable transform $\mathcal \Psi(\mathbf)$ s.t. $\mathcal \Psi( \mathbf x) \sim \mathcal N(0, \mathbf I)$.

$$\mathcal G:\mathbf x^{(k+1)}=\mathbf R_{(k)}\cdot \mathbf \Psi_{(k)}\left( \mathbf x^{(k)} \right)$$

where:
* $\mathbf \Psi_{(k)}$ is the marginal Gaussianization of each dimension of $\mathbf x_{(k)}$ for the corresponding iteration.
* $\mathbf R_{(k)}$ is the rotation matrix for the marginally Gaussianized variable $\mathbf \Psi_{(k)}\left( \mathbf x_{(k)} \right)$



---
### Marginal (Univariate) Gaussianization

This transformation is the $\mathcal \Psi_\theta$ step for the RBIG algorithm.

In theory, to go from any distribution to a Gaussian distribution, we just need to

To go from $\mathcal P \rightarrow \mathcal G$

1. Convert the Data to a Normal distribution $\mathcal U$
2. Apply the CDF of the Gaussian distribution $\mathcal G$
3. Apply the inverse Gaussian CDF


So to break this down even further we need two key components

#### Marginal Uniformization

We have to estimate the PDF of the marginal distribution of $\mathbf x$. Then using the CDF of that estimated distribution, we can compute the uniform

$u=U_k(x^k)=\int_{-\infty}^{x^k}\mathcal p(x^k)dx^k$

This boils down to estimating the histogram of the data distribution in order to get some probability distribution. I can think of a few ways to do this but the simplest is using the histogram function. Then convert it to a [scipy stats rv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html) where we will have access to functions like pdf and cdf. One nice trick is to add something to make the transformation smooth to ensure all samples are within the boundaries.

* Example Implementation - [https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/univariate.py]()

From there, we just need the CDF of the univariate function $u(\mathbf x)$. We can just used the ppf function (the inverse CDF / erf) in scipy

#### Gaussianization of a Uniform Variable

Let's look at the first step: Marginal Uniformization. There are a number of ways that we can do this.

To go from $\mathcal G \rightarrow \mathcal U$

---
### Linear Transformation

This is the $\mathcal R_\theta$ step in the RBIG algorithm. We take some data $\mathbf x_i$ and apply some rotation to that data $\mathcal R_\theta (\mathbf x_i)$. This rotation is somewhat flexible provided that it follows a few criteria:

* Orthogonal
* Orthonormal
* Invertible

So a few options that have been implemented include:

* Independence Components Analysis (ICA)
* Principal Components Analysis (PCA)
* Random Rotations (random)

We would like to extend this framework to include more options, e.g. 

* Convolutions (conv)
* Orthogonally Initialized Components (dct)


The whole transformation process goes as follows:

$$\mathcal P \rightarrow \mathbf W \cdot \mathcal P \rightarrow U \rightarrow G$$

Where we have the following spaces:

* $\mathcal P$ - the data space for $\mathcal X$.
* $\mathbf W \cdot \mathcal P$ - The transformed space.
* $\mathcal U$ - The Uniform space.
* $\mathcal G$ - The Gaussian space.


