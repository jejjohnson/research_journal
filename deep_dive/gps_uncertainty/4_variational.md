# Uncertain Inputs GPs - Variational Strategies


This post is a follow-up from my previous post where I walk through the literature talking about the different strategies of account for input error in Gaussian processes. In this post, I will be discussing how we can use variational strategies to account for input error.

---
## Posterior Approximations

What links all of the strategies from uncertain GPs is how they approach the problem of uncertain inputs: approximating the posterior distribution. The methods that use moment matching on stochastic trial points are all using various strategies to construct some posterior approximation. They define their GP model first and then approximate the posterior by using some approximate scheme to account for uncertainty. The NIGP however does change the model which is a product of the Taylor series expansion employed. From there, the resulting posterior is either evaluated or further approximated. My method actually is related because I also avoid changing the model and just attempt to approximate the posterior predictive distribution by augmenting the predictive variance function only (**???**).

This approach has similar strategies that stem from the wave of methods that appeared using variational inference (VI). VI consists of creating a variational approximation of the posterior distribution $q(\mathbf u)\approx \mathcal{P}(\mathbf x)$. Under some assumptions and a baseline distribution for $q(\mathbf u)$,  we can try to approximate the complex distribution $\mathcal{P}(\mathbf x)$ by minimizing the distance between the two distributions, $D\left[q(\mathbf u)||\mathcal{P}(\mathbf x)\right]$. Many practioners believe that approximating the posterior and not the model is the better option when doing Bayesian inference; especially for large data ([example blog](https://www.prowler.io/blog/sparse-gps-approximate-the-posterior-not-the-model), [VFE paper]()). The variational family of methods that are common for GPs use the Kullback-Leibler (KL) divergence criteria between the GP posterior approximation $q(\mathbf u)$ and the true GP posterior $\mathcal{P}(\mathbf x)$. From the literature, this has been extended to many different problems related to GPs for regression, classification, dimensionality reduction and more.

---
## Model

Let's build up the GP model from the variational inference perspective. We have the same GP prior as the standard GP regression model:

$$\mathcal{P}(f) \sim \mathcal{GP}\left(\mathbf m_\theta, \mathbf K_\theta  \right)$$

We have the same GP likelihood which stems from the relationship between the inputs and the outputs:

$$y = f(\mathbf x) + \epsilon_y$$

$$\mathcal{P}(y|f, \mathbf{x}) = \prod_{i=1}^{N}\mathcal{P}\left(y_i| f(\mathbf x_i) \right) \sim \mathcal{N}(f, \sigma_y^2\mathbf I)$$

Now we just need an variational approximation to the GP posterior:

$$q(f) = \mathcal{GP}\left( \mu, \nu^2 \right) $$

where $q(f) \approx  \mathcal{P}(f|y, \mathbf X)$.



$\mu$ and $\nu^2$ are functions that depend on the augmented space $\mathbf Z$ and possibly other parameters. Now, we can actually choose any $\mu$ and $\nu^2$ that we want. Typically people pick this to be Gaussian distributed which is augmented by some variable space $\mathcal{Z}$ with kernel functions to move us between spaces by a joint distribution; for example:

$$\mu(\mathbf x) = \mathbf k(\mathbf{x, Z})\mathbf{k(Z,Z)}^{-1}\mathbf m$$
$$\nu^2(\mathbf x) = \mathbf k(\mathbf{x,x}) - \mathbf k(\mathbf{x, Z})\left( \mathbf{k(Z,Z)}^{-1}  - \mathbf{k(Z,Z)}^{-1} \Sigma \mathbf{k(Z,Z)}^{-1}\right)\mathbf k(\mathbf{x, Z})^{-1}$$

where $\theta = \{ \mathbf{m, \Sigma, Z} \}$ are all variational parameters. This formulation above is just the end result of using augmented values by using variational compression (see [here]() for more details). In the end, all of these variables can be adjusted to reduce the KL divergence criteria KL$\left[ q(f)||\mathcal{P}(f|y, \mathbf X)\right]$.

There are some advantages to the approximate-prior approach for example:

* The approximation is non-parametric and mimics the true posterior.
* As the number of inducing points grow, we arrive closer to the real distribution
* The pseudo-points $\mathbf Z$ and the amount are also parameters which can protect us from overfitting.
* The predictions are clear as we just need to evaluate the approximate GP posterior.

---
### Evidence Lower Bound (ELBO)

In VI strategies, we never get an explicit function that will lead us to the best variational approximation but we can come close; we can come up with an upper bound. 
Traditional marginal likelhood (evidence) function that we're given is given by:

$$\underbrace{\mathcal{P}(y|\mathbf x, \theta)}_{\text{Evidence}}=\int_f \underbrace{\mathcal{P}(y|f, \mathbf x, 
\theta)}_{\text{Likelihood}} \cdot \underbrace{\mathcal{P}(f|\mathbf x, \theta)}_{\text{GP Prior}}df$$

where:

* $\mathcal{P}(y|f, \mathbf x, \theta)=\mathcal{N}(y|f, \sigma_n^2\mathbf{I})$
* $\mathcal{P}(f|\mathbf x, \theta)=\mathcal{N}(f|\mu, K_\theta)$

In this case we are marginalizing by the latent functions $f$'s. But we no longer consider $\mathbf x$ to be uncertain with some probability distribution. Also, we do not want some MAP estimation; we want a fully Bayesian approach. So the only thing to do is marginalize out the $\mathbf x$'s. In doing so we get:

$$\mathcal{P}(y| \theta)=\int_f\int_\mathcal{X} \mathcal{P}(y|f, \mathbf x, 
\theta)\cdot\mathcal{P}(f|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x)\cdot df \cdot d\mathbf{x}$$

We can rearrange this equation to change notation:

$$\mathcal{P}(y| \theta)=\int_\mathcal{X} 
\underbrace{\left[ \int_f \mathcal{P}(y|f, \mathbf x, 
\theta)\cdot\mathcal{P}(f|\mathbf x, \theta)\cdot df\right]}_{\text{Evidence}}
\cdot \underbrace{\mathcal{P}(\mathbf x)}_{\text{Prior}} \cdot d\mathbf{x}$$

where we find that that term is simply the same term as the original likelihood, $\mathcal{P}(y|\mathbf x, \theta)$. So our new simplified equation is:

$$\mathcal{P}(y| \theta)=\int_\mathcal{X} \mathcal{P}(y|\mathbf x, \theta) \cdot \mathcal{P}(\mathbf x) \cdot d\mathbf{x}$$

where we have effectively marginalized out the $f$'s. We already know that it's difficult to propagate the $\mathbf x$'s through the nonlinear functions $\mathbf K^{-1}$ and $|$det $\mathbf K|$ (see previous doc for examples). So using the VI strategy, we introduce a new variational distribution $q(\mathbf x)$ to approximate the posterior distribution $\mathcal{P}(\mathbf x| y)$. The distribution is normally chosen to be Gaussian:

$$q(\mathbf x) = \prod_{i=1}^{N}\mathcal{N}(\mathbf x|\mathbf \mu_z, \mathbf \Sigma_z)$$

So at this point, we are interested in trying to find a way to measure the difference between the approximate distribution $q(\mathbf x)$ and the true posterior distribution $\mathcal{P} (\mathbf x)$. Using the standard derivation for the ELBO, we arrive at the final formula:


$$\mathcal{F}(q)=\mathbb{E}_{q(\mathbf x)}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right] - \text{D}_\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x) \right]$$

If we optimize $\mathcal{F}$ with respect to $q(\mathbf x)$, the KL is minimized and we just get the likelihood. As we've seen before, the likelihood term is still problematic as it still has the nonlinear portion to propagate the $\mathbf x$'s through. So that's nothing new and we've done nothing useful. If we introduce some special structure in $q(f)$ by introducing sparsity, then we can achieve something useful with this formulation.
 But through augmentation of the variable space with $\mathbf u$ and $\mathbf Z$ we can bypass this problem. The second term is simple to calculate because they're both chosen to be Gaussian.


---
## Uncertain Inputs

So how does this relate to uncertain inputs exactly?

$$\begin{aligned}
y &= f(x) + \epsilon_y \\
x &= z + \epsilon_x \\
\end{aligned}$$

where:

* $z$ - noise-corrupted training inputs
* $x$ - latent variables
  * $\epsilon_x \sim \mathcal{N}(0, \Sigma_x)$
* $f(x)$ - function with noisy-inputs
* $y$ - noise-corrupted outputs
  * $\epsilon_y \sim \mathcal{N}(0, \sigma^2_y)$


#### Latent Variables - Prior Distribution 

We can directly assume a prior distribution for the latent variables that depend on the noisy observations.

$$\mathcal{P}(\mathbf X|\mathbf Z) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |\mathbf z_i,\mathbf \Sigma_\mathbf{x_i})$$

We will have a new variational bound now:

$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf X)} - \text{KL}\left( q(\mathbf X) || \mathcal{P}(\mathbf{X|Z}) \right)$$
π
#### Variational Back-Constraint

We will introduce a variational constraint to encode the input uncertainty directly into the approximate posterior. 

$$q(\mathbf X|\mathbf Z) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |\mathbf z_i,\mathbf \Sigma_\mathbf{z_i})$$

We will have a new variational bound now:

$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf{X|Z})} - \text{KL}\left( q(\mathbf{X|Z}) || \mathcal{P}(\mathbf{X}) \right)$$

---
---
## Supplementary Material

### Jensen's Inequality

This theorem is one of those [sleeper theorems](https://www.johndcook.com/blog/2012/12/10/sleeper-theorems/) which comes up in a big way in many machine learning problems. I've seen it mostly in the context of optimization and variational inference. 

The Jensen inequality theorem states that for a convex function $f$, 

$$\mathbb{E} [f(x)] \geq f(\mathbb{E}[x])$$

A convex function (or concave up) is when there exists a minimum to that function. If we take two points on any part of the graph and draw a line between them, we will be above or at (as a limit) the minimum point of the graph. We can flip the signs for a concave function. But we want the convex property because then it means it has a minimum value and this is useful for minimization strategies. Recall from Calculus class 101: let's look at the function $f(x)=\log x$.

We can use the second derivative test to find out if a function is convex or not. If $f'(x) \geq 0$ then it is concave up (or convex). I'll map out the derivatives below:

$$f'(x) = \frac{1}{x}$$
$$f''(x) = -\frac{1}{x^2}$$

You'll see that $-\frac{1}{x^2}\leq 0$ for $x \in [0, \infty)$. This means that $\log x$ is a concave function. So, the solution to this if we want a convex function is to take the negative $\log$ (which adds intuition as to why we typically take the negative log likelihood of many functions).

**Note on Variational Inference**:

Typically in the VI literature, they add this Jensen inequality property in order to come up with the Evidence Lower Bound (ELBO). But I never understood how it worked because I didn't know if they wanted the convex or concave. If we think of the loss function of the likelihood $\mathcal{L}(\theta)$ and the ELBO $\mathcal{F}(q, \theta)$. Take a look at the figure from

<p align="center">
  <img src="pics/elbo_inequality.png" alt="drawing" width="400"/>
</p>

**Figure:** Showing

Typically we use the $\log$ function when it's used 


**Resources**

* [Computational Statistics](http://people.duke.edu/~ccc14/sta-663-2016/14_ExpectationMaximization.html)
* [Blog](http://www.colaberry.com/jensens-inequality-that-guarantees-convergence-of-the-em-algorithm/)
* [Sleeper Theorems](https://www.johndcook.com/blog/2012/12/10/sleeper-theorems/)
* [DIT Package](https://dit.readthedocs.io/en/latest/measures/divergences/jensen_shannon_divergence.html)
* Ox Educ - [Intuition](https://www.youtube.com/watch?v=HfCb1K4Nr8M) | [Proof](https://www.youtube.com/watch?v=10xgmpG_uTs)
* MIT OpenCourseWare - [Intro Prob.](https://www.youtube.com/watch?v=GDJFLfmyb20) | [Inequalitiese, Convergence and Weak Law of Large Numbers](https://ocw.mit.edu/resources/res-6-012-introduction-to-probability-spring-2018/part-ii-inference-limit-theorems/)

---
### [Identity Trick](https://www.shakirm.com/slides/MLSS2018-Madrid-ProbThinking.pdf)

I think the slides from the MLSS 2018 meeting is the only place that I have encountered anyone actually explicitly mentioning this Identity trick.

Given an integral problem: 

$$p(x) = \int p(x|z)p(z)dz$$ 

I can multiply by an arbitrary distribution which is equivalent to 1.

$$p(x)=\int p(x|z) p(z) \frac{q(z)}{q(z)}dz$$

Then I can regroup and reweight the integral

$$p(x) = \int p(x|z)\frac{p(z)}{q(z)}q(z)dz$$

This results in a different expectation that we initially had

$$p(x) = \underset{q(z)}{\mathbb{E}}\left[ p(x|z)\frac{p(z)}{q(z)} \right]$$

Examples:

* Importance Sampling
* Manipulate Stochastic gradients
* Derive Probability bounds
* RL for policy corrections

---
### Variational Inference in a Nutshell



---
--- 
## Resources


#### Important Papers

These are the important papers that helped me understand what was going on throughout the learning process.


#### Summary Thesis

Often times the papers that people publish in conferences in Journals don't have enough information in them. Sometimes it's really difficult to go through some of the mathematics that people put  in their articles especially with cryptic explanations like "it's easy to show that..." or "trivially it can be shown that...". For most of us it's not easy nor is it trivial. So I've included a few thesis that help to explain some of the finer details. I've arranged them in order starting from the easiest to the most difficult.


* [Non-Stationary Surrogate Modeling with Deep Gaussian Processes](https://lib.ugent.be/fulltxt/RUG01/002/367/115/RUG01-002367115_2017_0001_AC.pdf) - Dutordoir (2016)
  * Chapter IV - Finding Uncertain Patterns in GPs
* [Bringing Models to the Domain: Deploying Gaussian Processes in the Biological Sciences](http://etheses.whiterose.ac.uk/18492/1/MaxZwiesseleThesis.pdf) - Zwießele (2017)
  * Chapter II (2.4, 2.5) - Sparse GPs, Variational Bayesian GPLVM
* [Deep GPs and Variational Propagation of Uncertainty](http://etheses.whiterose.ac.uk/9968/1/Damianou_Thesis.pdf) - Damianou (2015)
  * Chapter IV - Uncertain Inputs in Variational GPs
  * Chapter II (2.1) - Lit Review




#### Talks

* Damianou - Bayesian LVM with GPs - [MLSS2015](http://gpss.cc/gpss15/talks/gpss_BGPLVMs.pdf)
* Lawrence - Deep GPs - [MLSS2019]()

