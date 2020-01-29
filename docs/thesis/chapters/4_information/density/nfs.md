# Normalizing Flows

---
## Density Destructor

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

---

## Survey of Literature

## Code Tutorials

* Building Prob Dist with TF Probability Bijector API - [Blog](https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/)
* https://www.ritchievink.com/blog/2019/10/11/sculpting-distributions-with-normalizing-flows/