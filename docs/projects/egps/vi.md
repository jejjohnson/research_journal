# Uncertain Inputs GPs - Variational Strategies


This post is a follow-up from my previous post where I walk through the literature talking about the different strategies of account for input error in Gaussian processes. In this post, I will be discussing how we can use variational strategies to account for input error.

--- 

- [Posterior Approximations](#posterior-approximations)
  - [Variational GP Model with Latent Inputs](#variational-gp-model-with-latent-inputs)
  - [Evidence Lower Bound (ELBO)](#evidence-lower-bound-elbo)
- [Uncertain Inputs](#uncertain-inputs)
  - [Case I - Strong Prior](#case-i---strong-prior)
  - [Case II - Regularized Strong Prior](#case-ii---regularized-strong-prior)
  - [Case III - Prior with Openness](#case-iii---prior-with-openness)
  - [Case IV - Bonus, Conservative Freedom](#case-iv---bonus-conservative-freedom)
- [Resources](#resources)
    - [Important Papers](#important-papers)
    - [Summary Thesis](#summary-thesis)
    - [Talks](#talks)


## Posterior Approximations

What links all of the strategies from uncertain GPs is how they approach the problem of uncertain inputs: approximating the posterior distribution. The methods that use moment matching on stochastic trial points are all using various strategies to construct some posterior approximation. They define their GP model first and then approximate the posterior by using some approximate scheme to account for uncertainty. The NIGP however does change the model which is a product of the Taylor series expansion employed. From there, the resulting posterior is either evaluated or further approximated. My method actually is related because I also avoid changing the model and just attempt to approximate the posterior predictive distribution by augmenting the predictive variance function only (**???**).

This approach has similar strategies that stem from the wave of methods that appeared using variational inference (VI). VI consists of creating a variational approximation of the posterior distribution $q(\mathbf u)\approx \mathcal{P}(\mathbf x)$. Under some assumptions and a baseline distribution for $q(\mathbf u)$,  we can try to approximate the complex distribution $\mathcal{P}(\mathbf x)$ by minimizing the distance between the two distributions, $D\left[q(\mathbf u)||\mathcal{P}(\mathbf x)\right]$. Many practioners believe that approximating the posterior and not the model is the better option when doing Bayesian inference; especially for large data ([example blog](https://www.prowler.io/blog/sparse-gps-approximate-the-posterior-not-the-model), [VFE paper]()). The variational family of methods that are common for GPs use the Kullback-Leibler (KL) divergence criteria between the GP posterior approximation $q(\mathbf u)$ and the true GP posterior $\mathcal{P}(\mathbf x)$. From the literature, this has been extended to many different problems related to GPs for regression, classification, dimensionality reduction and more.

---

### Variational GP Model with Latent Inputs


**Posterior Distribution:**
$$p(Y) = \int_{\mathcal X} p(Y|X) P(X) dX$$

**Derive the Lower Bound** (w/ Jensens Inequality):

$$\log p(Y) = \log \int_{\mathcal X} p(Y|X) P(X) dX$$

**importance sampling/identity trick**

$$ = \log \int_{\mathcal F} p(Y|X) P(X) \frac{q(X)}{q(X)}dF$$

**rearrange to isolate**: $p(Y|X)$ and shorten notation to $\langle \cdot \rangle_{q(X)}$.

$$= \log \left\langle  \frac{p(Y|X)p(X)}{q(X)} \right\rangle_{q(X)}$$

**Jensens inequality**

$$\geq \left\langle \log \frac{p(Y|X)p(X)}{q(X)} \right\rangle_{q(X)}$$

**Split the logs**


$$\geq \left\langle \log p(Y|X) + \log \frac{p(X)}{q(X)} \right\rangle_{q(X)}$$

**collect terms**

$$\mathcal{L}_{2}(q)=\left\langle \log p(Y|X)\right\rangle_{q(F)} - D_{KL} \left( q(X) || p(X)\right) $$

**plug in other bound**

$$\mathcal{L}_{2}(q)=\left\langle \mathcal{L}_{1}(q)\right\rangle_{q(F)} - D_{KL} \left( q(X) || p(X)\right) $$


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

So how does this relate to uncertain inputs exactly? Let's look again at our problem setting.

$$\begin{aligned}
y &= f(x) + \epsilon_y \\
x &\sim \mathcal{N}(\mu_x, \Sigma_x) \\
\end{aligned}$$

where:

* $y$ - noise-corrupted outputs which have a noise parameter characterized by $\epsilon_y \sim \mathcal{N}(0, \sigma^2_y)$
* $f(x)$ - is the standard GP function
* $x$ - "latent variables" but we assume that the come from a normal distribution, $x \sim \mathcal{N}(\mu_x, \Sigma_x)$ where you have some observations $\mu_x$ but you also have some prior uncertainty $\Sigma_x$ that you would like to incorporate.

Now the ELBO that we want to minimize has the following form:

$$\mathcal{F}(q)=\mathbb{E}_{q(\mathbf x | m_{p_z}, S_{p_z})}\left[ \log \mathcal{P}(y|\mathbf x, \theta) \right] - \text{D}_\text{KL}\left[ q(\mathbf x | m_{p_z}, S_{p_z}) || \mathcal{P}(\mathbf x | m_{p_x}, S_{p_x}) \right]$$

Notice that I have expanded the parameters for $p(X)$ and $q(X)$ so that we are clear about where the parameters. We would like to figure out a way to incorporate our uncertainties in the $m_{p_x}$, $S_{p_x}$, $m_{p_z}$, and $S_{p_z}$. The author had two suggestions about how to account for noise in the inputs but the original formulation assumed that these parameters were unknown. In my problem setting, we know that there is noise in the inputs so the problems that the original formulations had will change. I will outline the formulations below for both known and unknown uncertainties.

---

### Case I - Strong Prior

**Prior**, $p(X)$

We can directly assume that we know the parameters for the prior distribution. So we let $\mu_x$ be our noisy observations and we let $\Sigma_x$ be our known covariance matrix for $X$. These parameters are fixed as we assume we know them and we would like this to be our prior. This seems to be the most natural as is where we have information and we would like to use it. So now our prior is in the form of:

$$\mathcal{P}(\mathbf X|\mu_x, \Sigma_x) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |\mathbf \mu_{\mathbf{x}_i},\mathbf \Sigma_\mathbf{x_i})$$

and this will be our regularization that we use for the KL divergence term.

**Variational**, $q(X)$

However, the variational parameters $m$ and $S$ are also important because that is being directly evaluated with the KL divergence term and the likelihood function $\log p(y|X, \theta)$. So ideally we would also like to constrain this as well if we know something. The first thing to do would be to fix them as well to what we know about our data, $m=\mu_x$ and $S_{p_z} = \Sigma_x$. So our prior for our variational distribution will be:

$$q(\mathbf X|\mu_x, \Sigma_x) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |\mathbf \mu_{\mathbf{x}_i},\mathbf \Sigma_\mathbf{x_i})$$


We now have our variational bound with the assumed parameters:

$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf X|\mu_x, \Sigma_x)} - \text{KL}\left( q(\mathbf X|\mu_x, \Sigma_x) || p(\mathbf X|\mu_x, \Sigma_x)  \right)$$

**Assessment**

So this is a very strong belief over our parameters. The KL divergence term will be zero because the distributions will be the same and we would have probably done some extra computations for no reason; we do need this in order to make the likelihood tractable but it doesn't make sense if we're not learning anything. But this is absolute because we have no reason to change anything. It's worth testing to see how this goes.

---

### Case II - Regularized Strong Prior

This is similar to the above statement, however we would be reducing the prior parameters to a standard 0 mean and 1 standard deviation. So our prior function will look like this

$$\mathcal{P}(\mathbf X|0, 1) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |0, 1)$$

This would allow the KL-Divergence criteria extra penalization for the loss function. It might change some of the parameters learned for the other regions of the ELBO. So our final loss function is:

$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf X|\mu_x, \Sigma_x)} - \text{KL}\left( q(\mathbf X|\mu_x, \Sigma_x) || p(\mathbf X|0, 1)  \right)$$

---

### Case III - Prior with Openness

The last option I think is the most interesting. It seems to incorporate the prior but also allow for some flexibility. In this option, 

We can pivot off of what the factor that the KL divergence term is simply a regularizer. So we could also go with a more conservative approach where we 

We will introduce a variational constraint to encode the input uncertainty directly into the approximate posterior. 

$$q(\mathbf X|\mathbf Z) = \prod_{i=1}^{N}\mathcal{N}(\mathbf{x}_i |\mathbf z_i,\mathbf \Sigma_\mathbf{z_i})$$

We will have a new variational bound now:

$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf X|\mu_x, S)} - \text{KL}\left( q(\mathbf X|\mu_x, S) || p(\mathbf X|\mu_x, \Sigma_x)  \right)$$

So the only free parameter in the variational bound is the actual variance of our inputs $S$ that stems from our variational distribution $q(X)$. Again, this seems like a nice balanced approach where we can incorporate prior information within our model but at the same time allow for some freedom to maybe find a better distribution to represent the noise. 

---

### Case IV - Bonus, Conservative Freedom

Ok, so the true last option would be to try and see how the algorithm thinks the results should be. We 


$$\mathcal{F}=\langle \log \mathcal{P}(\mathbf{Y|X}) \rangle_{q(\mathbf{X|Z})} - \text{KL}\left( q(\mathbf{X|Z}) || \mathcal{P}(\mathbf{X}) \right)$$



<center>

|          Options          | $m_{p_x}$ | $S_{p_x}$  | $m_{p_z}$ | $S_{p_z}$  |
| :-----------------------: | :-------: | :--------: | :-------: | :--------: |
|         No Prior          |  $\mu_x$  |     1      |  $\mu_x$  |    S_z     |
| Strong Conservative Prior |  $\mu_x$  |     1      |  $\mu_x$  | $\Sigma_x$ |
|       Strong Prior        |  $\mu_x$  | $\Sigma_x$ |  $\mu_x$  | $\Sigma_x$ |
|      Bayesian Prior       |  $\mu_x$  | $\Sigma_x$ |  $\mu_x$  |    S_z     |


</center>

**Caption**: Summary of Options

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

