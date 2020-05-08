# Gaussianization

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Notebooks:
  * [1D Gaussianization](https://colab.research.google.com/drive/1C-hP2XCii-DLwLmK1wyvET095ZLQTdON)
  * 

---

- [Why Gaussianization?](#why-gaussianization)
- [Main Idea](#main-idea)
  - [Loss Function](#loss-function)
    - [Negentropy](#negentropy)
  - [Methods](#methods)
    - [Projection Pursuit](#projection-pursuit)
    - [Gaussianization](#gaussianization)
- [Rotation-Based Iterative Gaussianization](#rotation-based-iterative-gaussianization)
- [Rotation-Based Iterative Gaussianization](#rotation-based-iterative-gaussianization-1)
  - [Marginal (Univariate) Gaussianization](#marginal-univariate-gaussianization)
    - [Marginal Uniformization](#marginal-uniformization)
    - [Gaussianization of a Uniform Variable](#gaussianization-of-a-uniform-variable)
  - [Linear Transformation](#linear-transformation)
- [Information Theory Measures](#information-theory-measures)
  - [Information](#information)
  - [Entropy](#entropy)
  - [Mutual Information](#mutual-information)
  - [KL-Divergence](#kl-divergence)
- [References](#references)

## Why Gaussianization?

> **Gaussianization**: Transforms multidimensional data into multivariate Gaussian data.

It is notorious that we say "assume our data is Gaussian". We do this all of the time in practice. It's because Gaussian data typically has nice properties, e.g. close-form solutions, dependence, etc(**???**). But as sensors get better, data gets bigger and algorithms get better, this assumption does not always hold. 

However, what if we could make our data Gaussian? If it were possible, then all of the nice properties of Gaussians can be used as our data is actually Gaussian. How is this possible? Well, we use a series of invertible transformations to transform our data $\mathcal X$ to the Gaussian domain $\mathcal Z$. The logic is that by independently transforming each dimension of the data followed by some rotation will eventually converge to a multivariate dataset that is completely Gaussian.

We can achieve statistical independence of data components. This is useful for the following reasons:

* We can process dimensions independently
* We can alleviate the curse of dimensionality
* We can tackle the PDF estimation problem directly
  * With PDF estimation, we can sample and assign probabilities. It really is the hole grail of ML models.
* We can apply and design methods that assume Gaussianity of the data
* Get insight into the data characteristics

---

## Main Idea

The idea of the Gaussianization frameworks is to transform some data distribution $\mathcal{D}$ to an approximate Gaussian distribution $\mathcal{N}$. Let $x$ be some data from our original distribution, $x\sim \mathcal{D}$ and $\mathcal{G}_{\theta}(\cdot)$ be the transformation to the Normal distribution $\mathcal{N}(0, \mathbf{I})$.
$$z=\mathcal{G}_{\theta}(x)$$

where:
* $x\sim$Data Distribtuion
* $\theta$ - Parameters of transformation 
* $\mathcal{G}$ - family of transformations from Data Distribution to Normal Distribution, $\mathcal{N}$.
* $z\sim\mathcal{N}(0, \mathbf{I})$


If the transformation is differentiable, we have a clear relationship between the input and output variables by means of the **change of variables transformation**:

$$
\begin{aligned}
p_\mathbf{x}(\mathbf{x}) 
&= p_\mathbf{y} \left[ \mathcal{G}_\theta(\mathbf{x})  \right] \left| \nabla_\mathbf{x} \mathcal{G}_\theta(\mathbf{x}) \right|
\end{aligned}
$$

where:

* $\left| \cdot \right|$ - absolute value of the matrix determinant
* $P_z \sim \mathcal{N}(0, \mathbf{I})$
* $\mathcal{P}_x$ - determined solely by the transformation of variables.


We can say that $\mathcal{G}_{\theta}$ provides an implicit density model on $x$ given the parameters $\theta$.



---




---

### Loss Function


as shown in the equation from the original [paper][1].

---

#### Negentropy


---

### Methods

---

#### Projection Pursuit


---

#### Gaussianization


---

## Rotation-Based Iterative Gaussianization

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
## Rotation-Based Iterative Gaussianization

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


---

## Information Theory Measures

<figure>
<center>
<img src="pics/rbig_it/Fig_1.png" width="500">
</center>
<center>
<figurecaption>
<b>Caption</b>: Information Theory measures in a nutshell.
</figurecaption>
</center>
</figure>




### Information



### Entropy


### Mutual Information

<figure>
<center>
<img src="pics/rbig_it/mi.png" alt="MI using RBIG" style="width:60%">
</center>
<center>
<figurecaption>
<b>Caption</b>: Schematic for finding the Mutual Information using using RBIG.
</figurecaption>
</center>
</figure>


$$
\begin{aligned}
I(\mathbf{x,y}) 
&=
T\left( \left[ 
    \mathcal{G}_\theta (\mathbf{X}), \mathcal{G}_\phi (\mathbf{Y})
    \right] \right)
\end{aligned}
$$

---

### KL-Divergence

<figure>
<center>
<img src="pics/rbig_it/kld.png" width="500">
</center>
<center>
<figurecaption>
<b>Caption</b>: Schematic for finding the KL-Divergence using using RBIG.
</figurecaption>
</center>
</figure>

Let $\mathcal{G}_\theta (\mathbf{X})$ be the Gaussianization of the variable $\mathbf{X}$ which is parameterized by $\theta$.

$$
\begin{aligned}
D_\text{KL}\left[ \mathbf{X||Y} \right]
&=
D_\text{KL}\left[ \mathbf{X ||\mathcal{G}_\theta(Y)} \right]
\\
&=
J\left[  \mathcal{G}_\theta (\mathbf{\hat{y}}) \right]
\end{aligned}
$$
## References


[1]: https://www.uv.es/lapeva/papers/Laparra11.pdf "Iterative Gaussianization: From ICA toRandom Rotations - Laparra et. al. - IEEE TNNs (2011)"