# Parametric Gaussianization

Key Papers:
* [Gaussianization](https://papers.nips.cc/paper/1856-gaussianization.pdf) - Chen and Gopinath (2001)
* [Iterative Gaussianization: from ICA to Random Rotations](https://www.uv.es/vista/vistavalencia/RBIG.htm) - Laparra et. al. (2010)

---

## Why Gaussianization?

> **Gaussianization**: Transform multidimensional data into multivariate Gaussian data.

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

$$\mathcal{P}_x(x)=
\mathcal{P}_{z}\left( \mathcal{G}_{\theta}(x) \right)
\left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right|
$$

where:

* $\left| \cdot \right|$ - absolute value of the matrix determinant
* $P_z \sim \mathcal{N}(0, \mathbf{I})$
* $\mathcal{P}_x$ - determined solely by the transformation of variables.


We can say that $\mathcal{G}_{\theta}$ provides an implicit density model on $x$ given the parameters $\theta$.


---
## History

---
## Cost Function

So, we have essentially described a model that transforms the data from the original data distribution $\mathcal{D}$ to the normal distribution $\mathcal{N}$ so now the question is: how well did we approximate the base distribution $\mathcal{N}$. We can use something called **negentropy** which is how far the transformed distribution is from the normal distribution. More concretely, it is the KLD between the transformed distribution, $P_y$ and the standard normal distribution, $\mathcal{N}\sim(0, \mathbf{I})$. We can write down the standard definition of entropy like so

$$D_{KLD}(P_z||\mathcal{N}(0, \mathbf{I}))=\int_{-\infty}^{\infty}\mathcal{P}_z(z) \log \frac{\mathcal{P}_z(z)}{\mathcal{N}(0, \mathbf{I})}dx$$

However, it might make a bit more sense intuitively to rewrite this equation in terms of expectations.

$$\mathcal{J}(\mathcal{P}_z)=\mathbb{E}_z\left[ \log \mathcal{P}_z(z) - \log \mathcal{N}(z)\right]$$

This basically says want the expected value between the probabilities of our approximate base distribution $\mathcal{P}_z(z)$ and the real base distribution $\mathcal{N}(z)$. We have the equation of $\mathcal{P}_x(x)$ in terms of the probability of the base distribution $\mathcal{P}_z(z)$ , so we can plug that into our negentropy $\mathcal{J}(\mathcal{P}_z)$ formulation

$$\mathcal{J}(\mathcal{P}_z)=\mathbb{E}_z\left[ \log \left( \mathcal{P}_x(x)\left| \frac{\partial z}{\partial x} \right|^{-1}\right) - \log \mathcal{N}(z)\right]$$

We can unravel the log probabilities to something much simpler:

$$\mathcal{J}(\mathcal{P}_z)=\mathbb{E}_z\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial z}{\partial x} \right| - \log \mathcal{N}(z)\right]$$

Now, it's difficult to compute the expectations in terms of the base distribution $z$. Instead let's make it factor of our data. We can do this by unravelling the $\mathbb{E}_z$

$$\mathcal{J}(\mathcal{P}_z)=\sum_{-\infty}^{\infty}\mathcal{P}_z(z)\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial z}{\partial x} \right| - \log \mathcal{N}(z)\right]$$

Again, we utilize the fact that we've done a change of variables which means we can rewrite the expectation in terms of the Data distribution:

$$\mathcal{J}(\mathcal{P}_z)=\sum_{-\infty}^{\infty}\mathcal{P}_{x}\left( x \right)
\left| \frac{\partial z}{\partial x} \right|^{-1}\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial z}{\partial x} \right| - \log \mathcal{N}(z)\right]$$

which means we can simplify this to be the expectation w.r.t. to the data distribution:

$$\mathcal{J}(\mathcal{P}_z)=
\mathbb{E}_x\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial z}{\partial x} \right| - \log \mathcal{N}(z)\right]$$

Now, to be more concrete about where our variables are coming from, we can substitute the $z=\mathcal{G}_{\theta}(x)$ into our negentropy formulation:

$$\mathcal{J}(\mathcal{P}_z)=
\mathbb{E}_x\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right| - \log \mathcal{N}(\mathcal{G}_{\theta}(x))\right]$$

So now when it comes to minimizing the loss function, we just need to take the derivative w.r.t. to the parameters $\theta$. All of our terms in this equation are dependent on the parameter $\theta$. 

$$\frac{\partial \mathcal{J}(\mathcal{P}_z)}{\partial \theta}=
\frac{\partial}{\partial \theta}
\mathbb{E}_x\left[ \log \mathcal{P}_x(x) - \log \left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right| - \log \mathcal{N}(\mathcal{G}_{\theta}(x))\right]$$


The derivative of an expectation of something is the same as the expectation of a derivative ($\frac{\partial}{\partial \theta}(\mathbb{E}_x[\cdot]=\mathbb{E}_x[\frac{\partial}{\partial \theta}(\cdot)]$) using the dominated convergence theorem ([stackoverflow](https://math.stackexchange.com/questions/217702/when-can-we-interchange-the-derivative-with-an-expectation)). So we can just take the derivative w.r.t. $\theta$ inside of the expectation

$$\frac{\partial \mathcal{J}(\mathcal{P}_z)}{\partial \theta}=

\mathbb{E}_x\left[ \frac{\partial}{\partial \theta}(\log \mathcal{P}_x(x)) - \frac{\partial}{\partial \theta} \left( \log \left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right|\right) - \frac{\partial}{\partial \theta} \left( \log \mathcal{N}(\mathcal{G}_{\theta}(x)) \right) \right]$$

Let's take it term by term. First of all, we can see that the $\log \mathcal{P}_x(x)$ has no parameters dependent upon $\theta$ so we can immediately cancel that term.

$$\frac{\partial \mathcal{J}(\mathcal{P}_z)}{\partial \theta}=

\mathbb{E}_x\left[ \cancel{\frac{\partial}{\partial \theta}(\log \mathcal{P}_x(x))} - \frac{\partial}{\partial \theta} \left( \log \left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right|\right) - \frac{\partial}{\partial \theta} \left( \log \mathcal{N}(\mathcal{G}_{\theta}(x)) \right) \right]$$

<span style="color:red">The second term</span> ??? [third term](https://stats.stackexchange.com/questions/154133/how-to-get-the-derivative-of-a-normal-distribution-w-r-t-its-parameters)


Practically speaking, this is a bit difficult to calculate. Instead we can do a procedure that measures how much more Gaussian the approximate base distribution has become as a result of the transformation $\mathcal{G}_{\theta}(x)$.

--
## In the Wild

* [Gaussianization for Fast and Accurate Inference from Cosmological Data](https://arxiv.org/pdf/1510.00019.pdf)
  > Nice formula for how to calculate the likelihood.

