# Related Methods

- [Deep Density Destructors](#deep-density-destructors)
  - [Main Idea](#main-idea)
- [Normalizing Flows](#normalizing-flows)
  - [Loss Function](#loss-function)
  - [Sampling](#sampling)
- [Choice of Transformations](#choice-of-transformations)
  - [Prior Distribution](#prior-distribution)
  - [Jacobian](#jacobian)
- [Resources](#resources)
    - [Best Tutorials](#best-tutorials)
- [Survey of Literature](#survey-of-literature)
  - [Neural Density Estimators](#neural-density-estimators)
  - [Deep Density Destructors](#deep-density-destructors-1)
- [Code Tutorials](#code-tutorials)
  - [Tutorials](#tutorials)
  - [Algorithms](#algorithms)
  - [RBIG Upgrades](#rbig-upgrades)
  - [Cutting Edge](#cutting-edge)
  - [Github Implementations](#github-implementations)


## Deep Density Destructors

### Main Idea


We can view the approach of modeling from two perspectives: constructive or destructive. A constructive process tries to learn how to build an exact sequence of transformations to go from $z$ to $x$. The destructive process does the complete opposite and decides to create a sequence of transforms from $x$ to $z$ while also remembering the exact transforms; enabling it to reverse that sequence of transforms.

We can write some equations to illustrate exactly what we mean by these two terms. Let's define two spaces: one is our data space $\mathcal X$ and the other is the base space $\mathcal Z$. We want to learn a transformation $f_\theta$ that maps us from $\mathcal X$ to $\mathcal Z$, $f : \mathcal X \rightarrow \mathcal Z$. We also want a function $G_\theta$ that maps us from $\mathcal Z$ to $\mathcal X$, $f : \mathcal Z \rightarrow \mathcal X$.

**TODO: Plot**

More concretely, let's define the following pair of equations:

$$z \sim \mathcal{P}_\mathcal{Z}$$
$$\hat x = \mathcal G_\theta (z)$$

This is called the generative step; how well do we fit our parameters such that $x \approx \hat x$. We can define the alternative step below:

$$x \sim \mathcal{P}_\mathcal{X}$$
$$\hat z = \mathcal f_\theta (x)$$

This is called the inference step: how well do we fit the parameters of our transformation $f_\theta$ s.t. $z \approx \hat z$. So there are immediately some things to notice about this. Depending on the method you use in the deep learning community, the functions $\mathcal G_\theta$ and $f_\theta$ can be defined differently. Typically we are looking at the class of algorithms where we want $f_\theta = \mathcal G_\theta^{-1}$. In this ideal scenario, we only need to learn one transformation instead of two. With this requirement, we can actually compute the likelihood values exactly. The likelihood of the value $x$ given the transformation $\mathcal G_\theta$ is given as:

$$\mathcal P_{\hat x}(x)=\mathcal P_{z} \left( \mathcal G_\theta (x) \right)\left| \text{det } \mathbf J_{\mathcal G_\theta} \right|$$

## Normalizing Flows

> *Distribution flows through a sequence of invertible transformations* - Rezende & Mohamed (2015)


We want to fit a density model $p_\theta(x)$ with continuous data $x \in \mathbb{R}^N$. Ideally, we want this model to:

* **Modeling**: Find the underlying distribution for the training data.
* **Probability**: For a new $x' \sim \mathcal{X}$, we want to be able to evaluate $p_\theta(x')$
* **Sampling**: We also want to be able to generate samples from $p_\theta(x')$.
* **Latent Representation**: Ideally we want this representation to be meaningful.


Let's assume that we can find some probability distribution for $\mathcal{X}$ but it's very difficult to do. So, instead of $p_\theta(x)$, we want to find some parameterized function $f_\theta(x)$ that we can learn.

$$x = f_\theta(x)$$

We'll define this as $z=f_\theta(x)$. So we also want $z$ to have certain properties. 

1. We want this $z$ to be defined by a probabilistic function and have a valid distribution $z \sim p_\mathcal{Z}(z)$
2. We also would prefer this distribution to be simply. We typically pick a normal distribution, $z \sim \mathcal{N}(0,1)$





 We begin with in initial distribution and then we apply a sequence of $L$ invertible transformations in hopes that we obtain something that is more expressive. This originally came from the context of Variational AutoEncoders (VAE) where the posterior was approximated by a neural network. The authors wanted to 

$$
\begin{aligned}
\mathbf{z}_L = f_L \circ f_{L-1} \circ \ldots \circ f_2 \circ f_1 (\mathbf{z}_0)
\end{aligned}
$$



### Loss Function

We can do a simple maximum-likelihood of our distribution $p_\theta(x)$. 

$$\underset{\theta}{\text{max}} \sum_i \log p_\theta(x^{(i)})$$

However, this expression needs to be transformed in terms of the invertible functions $f_\theta(x)$. This is where we exploit the rule for the change of variables. From here, we can come up with an expression for the likelihood by simply calculating the maximum likelihood of the initial distribution $\mathbf{z}_0$ given the transformations $f_L$. 



$$
\begin{aligned}
p_\theta(x) = p_\mathcal{Z}(f_\theta(x)) \left| \frac{\partial f_\theta(x)}{\partial x} \right|
\end{aligned}
$$

So now, we can do the same maximization function but with our change of variables formulation:

$$
\begin{aligned}
\underset{\theta}{\text{max}} \sum_i \log p_\theta(x^{(i)}) &= 
\underset{\theta}{\text{max}} \sum_i \log p_\mathcal{Z}\left(f_\theta(x^{(i)})\right) +
\log \left| \frac{\partial f_\theta (x^{(i)})}{\partial x} \right|
\end{aligned}
$$

And we can optimize this using stochastic gradient descent (SGD) which means we can use all of the autogradient and deep learning libraries available to make this procedure relatively painless.

### Sampling

If we want to sample from our base distribution $z$, then we just need to use the inverse of our function. 

$$x = f_\theta^{-1}(z)$$

where $z \sim p_\mathcal{Z}(z)$. Remember, our $f_\theta(\cdot)$ is invertible and differentiable so this should be no problem.


---


$$
\begin{aligned}
q(z') = q(z) \left| \frac{\partial f}{\partial z} \right|^{-1}
\end{aligned}
$$

or the same but only in terms of the original distribution $\mathcal{X}$


We can make this transformation a bit easier to handle empirically by calculating the Log-Transformation of this expression. This removes the inverse and introduces a summation of each of the transformations individually which gives us many computational advantages.

$$
\begin{aligned}
\log q_L (\mathbf{z}_L) = \log q_0 (\mathbf{z}_0) - \sum_{l=1}^L \log \left| \frac{\partial f_l}{\partial \mathbf{z}_l} \right|
\end{aligned}
$$

So now, our original expression with $p_\theta(x)$ can be written in terms of $z$.



TODO: Diagram with plots of the Normalizing Flow distributions which show the direction for the idea.

In order to train this, we need to take expectations of the transformations.

$$
\begin{aligned}
\mathcal{L}(\theta) &= 
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log p(\mathbf{x,z}_L)\right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} \left[ \log q_0(\mathbf{z}_0) \right] -
\mathbb{E}_{q_0(\mathbf{z}_0)} 
\left[ \sum_{l=1}^L \log \text{det}\left| \frac{\partial f_l}{\partial \mathbf{z}_k} \right| \right]
\end{aligned}
$$



## Choice of Transformations

The main thing that many of the communities have been looking into is how one chooses the aspects of the normalizing flow: the prior distribution and the Jacobian. 


### Prior Distribution

This is very consistent across the literature: most people use a fully-factorized Gaussian distribution. Very simple.

### Jacobian

This is the area of the most research within the community. There are many different complicated frameworks but almost all of them can be put into different categories for how the Jacobian is constructed.

## Resources

#### Best Tutorials

* [Flow-Based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html) - Lilian Weng
  > An excellent blog post for Normalizing Flows. Probably the most thorough introduction available.
* [Flow Models](https://docs.google.com/presentation/d/1WqEy-b8x-PhvXB_IeA6EoOfSTuhfgUYDVXlYP8Jh_n0/edit#slide=id.g7d4f9f0446_0_43) - [Deep Unsupervised Learning Class](https://sites.google.com/view/berkeley-cs294-158-sp20/home), Spring 2010 
* [Normalizing Flows: A Tutorial](https://docs.google.com/presentation/d/1wHJz9Awhlp-PWLZGWJKzF66gzvqdSrhknb-iLFJ1Owo/edit#slide=id.p) - Eric Jang



---

## Survey of Literature

---

### Neural Density Estimators

### Deep Density Destructors

## Code Tutorials

* Building Prob Dist with TF Probability Bijector API - [Blog](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/)
* https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/





### Tutorials

* RealNVP - [code I](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day3/nf/nf-solution.ipynb)
* [Normalizing Flows: Intro and Ideas](https://arxiv.org/pdf/1908.09257.pdf) - Kobyev et. al. (2019)


### Algorithms

*


### RBIG Upgrades

* Modularization
  * [Lucastheis](https://github.com/lucastheis/mixtures)
  * [Destructive-Deep-Learning](https://github.com/davidinouye/destructive-deep-learning/tree/master)
* TensorFlow
  * [NormalCDF](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/normal_cdf.py)
  * [interp_regular_1d_grid](https://www.tensorflow.org/probability/api_docs/python/tfp/math/interp_regular_1d_grid)
  * [IT w. TF](https://nbviewer.jupyter.org/github/adhiraiyan/DeepLearningWithTF2.0/blob/master/notebooks/03.00-Probability-and-Information-Theory.ipynb)


### Cutting Edge

* Neural Spline Flows - [Github](https://github.com/bayesiains/nsf)
  * **Complete** | PyTorch
* PointFlow: 3D Point Cloud Generations with Continuous Normalizing Flows - [Project](https://www.guandaoyang.com/PointFlow/)
  * PyTorch
* [Conditional Density Estimation with Bayesian Normalising Flows](https://arxiv.org/abs/1802.04908) | [Code](https://github.com/blt2114/CDE_with_BNF)

### Github Implementations

* [Bayesian and ML Implementation of the Normalizing Flow Network (NFN)](https://github.com/siboehm/NormalizingFlowNetwork)| [Paper](https://arxiv.org/abs/1907.08982)
* [NFs](https://github.com/ktisha/normalizing-flows)| [Prezi](https://github.com/ktisha/normalizing-flows/blob/master/presentation/presentation.pdf)
* [Normalizing Flows Building Blocks](https://github.com/colobas/normalizing-flows)
* [Neural Spline Flow, RealNVP, Autoregressive Flow, 1x1Conv in PyTorch](https://github.com/tonyduan/normalizing-flows)
* [Clean Refactor of Eric Jang w. TF Bijectors](https://github.com/breadbread1984/FlowBasedGenerativeModel)
* [Density Estimation and Anomaly Detection with Normalizing Flows](https://github.com/rom1mouret/anoflows)