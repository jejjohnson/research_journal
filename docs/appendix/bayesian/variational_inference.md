# Variational Inference

---
## Motivations

Variational inference is the most scalable inference method the machine learning community has (as of 2019).

**Tutorials**

* https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/
* https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/
* 

* * 

---
## ELBO - Derivation

Let's start with the marginal likelihood function.

$$\mathcal{P}(y| \theta)=\int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot d\mathbf{x}$$

where we have effectively marginalized out the $f$'s. We already know that it's difficult to propagate the $\mathbf x$'s through the nonlinear functions $\mathbf K^{-1}$ and $|$det $\mathbf K|$ (see previous doc for examples). So using the VI strategy, we introduce a new variational distribution $q(\mathbf x)$ to approximate the posterior distribution $\mathcal{P}(\mathbf x| y)$. The distribution is normally chosen to be Gaussian:

$$q(\mathbf x) = \prod_{i=1}^{N}\mathcal{N}(\mathbf x|\mathbf \mu_z, \mathbf \Sigma_z)$$

So at this point, we aree interested in trying to find a way to measure the difference between the approximate distribution $q(\mathbf x)$ and the true posterior distribution $\mathcal{P} (\mathbf x)$. Using some algebra, let's take the log of the marginal likelihood (evidence):

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot d\mathbf x$$

So now we are going to use the some tricks that you see within almost every derivation of the VI framework. The first one consists of using the Identity trick. This allows us to change the expectation to incorporate the new variational distribution $q(\mathbf x)$. We get the following equation:

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot \frac{q(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x$$

Now that we have introduced our new variational distribution, we can regroup and reweight our expectation. Because I know what I want, I get the following:

$$\log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot  q(\mathbf x) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x$$

Now with Jensen's inequality, we have the relationship $f(\mathbb{E}[x]) \leq \mathbb{E} [f(x)]$. We would like to put the $\log$ function inside of the integral. Jensen's inequality allows us to do this. If we let $f(\cdot)= \log(\cdot)$ then we get the Jensen's equality for a concave function, $f(\mathbb{E}[x]) \geq \mathbb{E} [f(x)]$. In this case if we match the terms to each component to the inequality, we have 

$$\log \cdot \mathbb{E}_\mathcal{q(\mathbf x)} \left[ \mathcal{P}(y|\mathbf x, \theta) \cdot  \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right] 
\geq  
\mathbb{E}_\mathcal{q(\mathbf x)} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right]$$

So now finally we have both terms in the inequality. Summarizing everything we have the following relationship:

$$log \mathcal{P}(y|\theta) = \log \int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot  q(\mathbf x) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \cdot d\mathbf x $$

$$
\log \mathcal{P}(y|\theta) \geq  \int_\mathcal{X} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right] q(\mathbf x) \cdot d\mathbf x
$$

I'm going to switch up the terminology just to make it easier aesthetically. I'm going to let $\mathcal{L}(\theta)$ be $\log \mathcal{P}(y|\theta)$ and $\mathcal{F}(q, \theta) \leq \mathcal{L}(\theta)$. So basically:

$$
\mathcal{L}(\theta) =\log \mathcal{P}(y|\theta) \geq  \int_\mathcal{X} \left[\log  \mathcal{P}(y|\mathbf x, \theta) \cdot \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} \right] q(\mathbf x) \cdot d\mathbf x = \mathcal{F}(q, \theta)
$$

With this simple change I can talk about each of the parts individually. Now using log rules we can break apart the likelihood and the quotient. The quotient will be needed for the KL divergence.

$$
\mathcal{F}(q) =
\underbrace{\int_\mathcal{X} q(\mathbf x) \cdot \log  \mathcal{P}(y|\mathbf x, \theta) \cdot d\mathbf x}_{\mathbb{E}_{q(\mathbf{x})}} +
\underbrace{\int_\mathcal{X} q(\mathbf x) \log  \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)}   \cdot d\mathbf x}_{\text{KL}}
$$



The punchline of this (after many calculated manipulations), is that we obtain an optimization equation $\mathcal{F}(\theta)$:

$$\mathcal{F}(q)=\mathbb{E}_{q(\mathbf x)}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right] - \text{D}_\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x) \right]$$

where:

* Approximate posterior distribution: $q(x)$
  * The best match to the true posterior $\mathcal{P}(y|\mathbf x, \theta)$. This is what we want to calculate.
* Reconstruction Cost: $\mathbb{E}_{q(\mathbf x)}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right]$
  * The expected log-likelihood measure of how well the samples from $q(x)$ are able to explain the data $y$.
* Penalty: $\text{D}_\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x) \right]$
  * Ensures that the explanation of the data $q(x)$ doesn't deviate too far from your beliefs $\mathcal{P}(x)$. (Okham's razor constraint)

**Source**: [VI Tutorial](https://www.shakirm.com/papers/VITutorial.pdf) - Shakir Mohamed

If we optimize $\mathcal{F}$ with respect to $q(\mathbf x)$, the KL is minimized and we just get the likelihood. As we've seen before, the likelihood term is still problematic as it still has the nonlinear portion to propagate the $\mathbf x$'s through. So that's nothing new and we've done nothing useful. If we introduce some special structure in $q(f)$ by introducing sparsity, then we can achieve something useful with this formulation.
 But through augmentation of the variable space with $\mathbf u$ and $\mathbf Z$ we can bypass this problem. The second term is simple to calculate because they're both chosen to be Gaussian.

 ### Comments on $q(x)$

* We have now transformed our problem from an integration problem to an optimization problem where we optimize for $q(x)$ directly. 
* Many people tend to simplify $q$ but we could easily write some dependencies on the data for example $q(x|\mathcal{D})$. 
* We can easily see the convergence as we just have to wait until the loss (free energy) reaches convergence.
* Typically $q(x)$ is a Gaussian whereby the variational parameters are the mean and the variance. Practically speaking, we could freeze or unfreeze any of these parameters if we have some prior knowledge about our problem.
* Many people say 'tighten the bound' but they really just mean optimization: modifying the hyperparameters so that we get as close as possible to the true marginal likelihood.


 ## Pros and Cons

 ### Why Variational Inference?

 * Applicable to all probabilistic models
 * Transforms a problem from integration to one of optimization
 * Convergence assessment
 * Principled and Scalable approach to model selection
 * Compact representation of posterior distribution
 * Faster to converge
 * Numerically stable
 * Modern Computing Architectures (GPUs)

### Why Not Variational Inference?

* Approximate posterior only
* Difficulty in optimization due to local minima
* Under-estimates the variance of posterior
* Limited theory and guarantees for variational mehtods

---
## Resources

* Tutorial Series - [Why?](https://chrisorm.github.io/VI-Why.html) | [ELBO](https://chrisorm.github.io/VI-ELBO.html) | [MC ELBO](https://chrisorm.github.io/VI-MC.html) | [Reparameterization](https://chrisorm.github.io/VI-reparam.html) | [MC ELBO unBias](https://chrisorm.github.io/VI-ELBO-MC-approx.html) | [MC ELBO PyTorch](https://chrisorm.github.io/VI-MC-PYT.html) | [Talk](https://chrisorm.github.io/pydata-2018.html)
* Blog Posts: Neural Variational Inference
  * [Classical Theory](http://artem.sobolev.name/posts/2016-07-01-neural-variational-inference-classical-theory.html)
  * [Scaling Up](http://artem.sobolev.name/posts/2016-07-04-neural-variational-inference-stochastic-variational-inference.html)
  * [BlackBox Mode](http://artem.sobolev.name/posts/2016-07-05-neural-variational-inference-blackbox.html)
  * [VAEs and Helmholtz Machines](http://artem.sobolev.name/posts/2016-07-11-neural-variational-inference-variational-autoencoders-and-Helmholtz-machines.html)
  * [Importance Weighted AEs](http://artem.sobolev.name/posts/2016-07-14-neural-variational-importance-weighted-autoencoders.html)
  * [Neural Samplers and Hierarchical VI](http://artem.sobolev.name/posts/2019-04-26-neural-samplers-and-hierarchical-variational-inference.html)
  * [Importance Weighted Hierarchical VI](http://artem.sobolev.name/posts/2019-05-10-importance-weighted-hierarchical-variational-inference.html) | [Video](https://www.youtube.com/watch?v=pdSu7XfGhHw&feature=youtu.be)

* Normal Approximation to the Posterior Distribution - [blog](http://bjlkeng.github.io/posts/normal-approximations-to-the-posterior-distribution/)

**Lower Bound**

* [Understaing the Variational Lower Bound](http://legacydirs.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf)
* [Deriving the Variational Lower Bound](http://paulrubenstein.co.uk/deriving-the-variational-lower-bound/)

**Summaries**

* [Advances in Variational Inference](https://arxiv.org/pdf/1711.05597.pdf)

**Presentations**

* [VI Shakir](https://www.shakirm.com/papers/VITutorial.pdf)
* Deisenroth - [VI](https://drive.google.com/file/d/1sAIF0rqgNbVbp7ZbuiS7kh96Yns04k1i/view) | [IT](https://drive.google.com/open?id=14WOcbwn011rJbFFsSbeoeuSxY4sMG4KY)
* [Bayesian Non-Parametrics and Priors over functions](https://www.doc.ic.ac.uk/~mpd37/teaching/ml_tutorials/2017-11-22-Ek-BNP-and-priors-over-functions.pdf)
* [here](https://filebox.ece.vt.edu/~s14ece6504/slides/Moran_I_ECE_6504_VB.pdf)


**Reviews**
* [From EM to SVI](http://krasserm.github.io/2018/04/03/variational-inference/)
* [Variational Inference](https://ermongroup.github.io/cs228-notes/inference/variational/)
* [VI- Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)
* [Tutorial on VI](http://www.robots.ox.ac.uk/~sjrob/Pubs/vbTutorialFinal.pdf)
* [VI w/ Code](https://zhiyzuo.github.io/VI/)
* [VI - Mean Field](https://blog.evjang.com/2016/08/variational-bayes.html)
* [VI Tutorial](https://github.com/philschulz/VITutorial)
* GMM
  * [VI in GMM](https://github.com/bertini36/GMM)
  * [GMM Pyro](https://mattdickenson.com/2018/11/18/gmm-python-pyro/) | [Pyro](http://pyro.ai/examples/gmm.html)
  * [GMM PyTorch](https://github.com/ldeecke/gmm-torch) | [PyTorch](https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html) | [PyTorchy](https://github.com/RomainSabathe/dagmm/blob/master/gmm.py)

**Code**


**Extensions**

* [Neural Samplers and Hierarchical Variational Inference](http://artem.sobolev.name/posts/2019-04-26-neural-samplers-and-hierarchical-variational-inference.html)


#### From Scratch

* Programming a Neural Network from Scratch - Ritchie Vink (2017) - [blog](https://www.ritchievink.com/blog/2017/07/10/programming-a-neural-network-from-scratch/)
* An Introduction to Probability and Computational Bayesian Statistcs - Eric Ma 0[Blog](https://ericmjl.github.io/essays-on-data-science/machine-learning/computational-bayesian-stats/)
* Variational Inference from Scratch - Ritchie Vink (2019) - [blog](https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/)
* Bayesian inference; How we are able to chase the Posterior - Ritchie Vink (2019) - [blog](Bayesian inference; How we are able to chase the Posterior)
* Algorithm Breakdown: Expectation Maximization - [blog](https://www.ritchievink.com/blog/2019/05/24/algorithm-breakdown-expectation-maximization/)
## Variational Inference

* Variational Bayes and The Mean-Field Approximation - Keng (2017) - [blog](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/)