Gaussian Processes



---
### Definition

The first thing to understand about GPs is that we are actively placing a distribution $\mathcal{P}(f)$ on functions $f$ where these functions can be infinitely long function values $f=[f_1, f_2, \ldots]$. A GP generalizes the multivariate Gaussian distribution to infinitely many variables.

> A GP is a collection of random variables $f_1, f_2, \ldots$, any finite number of which is Gaussian distributed.

> A GP defines a distribution over functions $p(f)$ which can be used for Bayesian regression. (Zhoubin)

Another nice definition is:

> **Gaussian Process**: Any set of function variables $\{f_n \}^{N}_{n=1}$ has a joint Gaussian distribution with mean function $m$. (Deisenroth)

The nice thing is that this is provided by a mean function $\mu$ and covariance matrix $\mathbf{K}$

---
### Bayesian Inference Problem

**Objective**


Let's have some data set, $\mathcal{D}= \left\{ (x_i, y_i)^N_{i=1} \right\}=(X,y)$

**Model**

$$
\begin{aligned}
y_i &= f(x_i) + \epsilon_i \\
f &\sim \mathcal{GP}(\cdot | 0, K) \\
\epsilon_i &\sim \mathcal{N}(\cdot | 0, \sigma^2)
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{P}(f_N) &= \int_{f_\infty}\mathcal{P}(f_N,f_\infty)df_\infty \\
&= \mathcal{N}(\mu_{f_N},\Sigma_{NN})
\end{aligned}
$$
The prior on $f$ is a GP distribution, the likelihood is Gaussian, therefore the posterior on $f$ is also a GP, 
$$
P(f|\mathcal{D}) \propto P(\mathcal{D}|f)P(f) = \mathcal{GP \propto G \cdot GP}
$$
So we can make predictions:
$$
P(y_*|x_*, \mathcal{D}) = \int P(y_*|x_*, \mathcal{D})P(f|\mathcal{D})df
$$
We can also do model comparison by way of the marginal likelihood (evidence) so that we can compare and tune the covariance functions
$$
P(y|X) = \int P(y|f,X)P(f)df
$$
**Bayesian Treatment**

So now how does this look in terms of the Bayes theorem:


$$
\begin{aligned}
\text{Posterior} &= \frac{\text{Likelihood}\cdot\text{Prior}}{\text{Evidence}}\\
p(f|X,y) &= \frac{p(y|f, X) \: p(f|X, \theta)}{p(y| X)} \\
\end{aligned}
$$


where:

* Prior: $p(f|X, \theta)=\mathcal{GP}(m_\theta, \mathbf{K}_\theta)$
* Likelihood (noise model): $p(y|f,X)=\mathcal{N}(y|f(x), \sigma_n^2\mathbf{I})$
* Marginal Likelihood (Evidence): $p(y|X)=\int_f p(y|f,X)p(f|X)df$
* Posterior: $p(f|X,y) = \mathcal{GP}(\mu_*, \mathbf{K}_*)$

---
## Gaussian Process Regression

We only need a few elements to define a Gaussian process in itself. Just a mean function $\mu$, a covariance matrix $\mathbf{K}_\theta$ and some data, $\mathcal{D}$.

```python
class GPR:
    def __init__(self, mu, kernel, X, y, noise_variance=1e-6):
        self.mu = mu
        self.kernel = kernel
        self.x_train = x_train
        self.y_train = y_train
        self.noise_variance = noise_variance
```

In the above script, we are

---

### GP Prior

This is the basis of the GP method. Under the assumption that we mentioned above:

$$
p(f|X, \theta)=\mathcal{GP}(m_\theta , \mathbf{K}_\theta)
$$
where:
* $m_\theta$ is a mean function
* $\mathbf{K}$ is a covariance function

We kind of treat these functions as a vector of function values up to infinity in theory $f=[f_1, f_2, \ldots]$. But in particular we look at the distribution over the function values, for example $f_i=f(x_i)$. So let's look at the joint distribution between $N$ function values $f_N$ and all other function values $f_\infty$. This is 'normally distributed' so we can write the joint distribution roughly as:
$$
\mathcal{P}(f_N, f_\infty)=\mathcal{N}
\left(\begin{bmatrix}
\mu_N \\ \mu_\infty
\end{bmatrix}, 
\begin{bmatrix}
\Sigma_{NN} & \Sigma_{N\infty} \\
\Sigma_{N\infty}^{\top} & \Sigma_{\infty\infty}
\end{bmatrix}\right)
$$
where $\Sigma_{NN}\in \mathbb{R}^{N\times N}$ and $\Sigma_{\infty\infty} \in \mathbb{R}^{\infty \times \infty}$ (or $m\rightarrow \infty$) to be more precise.

So again, any marginal distribution of a joint Gaussian distribution is still a Gaussian distribution. So if we integrate over all of the functions from the infinite portion, we get:

We can even get more specific and split the $f_N$ into training $f_{\text{train}}$ and testing $f_{\text{test}}$. It's simply a matter of manipulating joint Gaussian distributions. So again, calculating the marginals:


$$\begin{aligned}
\mathcal{P}(f_{\text{train}}, f_{\text{test}})
 &= \int_{f_\infty}\mathcal{P}(f_{\text{train}}, f_{\text{test}},f_\infty)df_\infty \\
&= \mathcal{N}
\left(\begin{bmatrix}
f_{\text{train}} \\ f_{\text{test}}
\end{bmatrix}, 
\begin{bmatrix}
\Sigma_{\text{train} \times \text{train}} & \Sigma_{\text{train} \times \text{test}} \\
\Sigma_{\text{train} \times \text{test}}^{\top} & \Sigma_{\text{test} \times \text{test}}
\end{bmatrix}\right)
\end{aligned}$$

and we arrive at a joint Gaussian distribution of the training and testing which is still normally distributed due to the marginalization.

Now, something a bit more practical, generally speaking when we program the sampling portion of the prior, we need data. The kernel function is as is and has already been defined with its appropriate parameters. Furthermore, we already have defined the mean function $\mu$ when we initialized the mean function above. So we just need to pass the function through the multivariate normal function along with the number of samples we would like to draw from the prior.


```python
def sample_prior(self, X, n_samples=1, random_state=None):
    """Draws random n_samples from the multivariate normal
    distribution:
    X ~ N(mu, K(X,X))
    
    Parameters
    ----------
    X : array, (N x D)
        The vector data use to draw the random samples from
        where N is the number of data and D is the dimension.
    
    n_samples : int (default = 1)
        The number of random samples to draw from the normal
        distribution
    
    random_state : int, (default = None)
        Needed if we want to specify the random seed used to 
        draw the random samples.

    Returns
    -------
    samples : array, (n_samples x N)
        The samples drawn from the multivariate normal
        distribution.
    """
    # calculate the covariance
    cov = self.kernel(X, X)

    # handle the random state
    rng = check_random_state(random_state)

    # draw samples from random multivariate normal
    samples = rng.multivariate_normal(self.mu.ravel(), cov, int(n_samples))

    return samples
```

---
### Likelihood (noise model)

$$
p(y|f,X)=\prod_{i=1}^{N}\mathcal{N}(y_i|f_i,\sigma_\epsilon^2)= \mathcal{N}(y|f(x), \sigma_\epsilon^2\mathbf{I}_N)
$$



This comes from our assumption as stated above from $y=f(x)+\epsilon$.

Alternative Notation:
* $y\sim \mathcal{N}(f, \sigma_n^2)$
* $\mathcal{N}(f, \sigma_n^2) = \prod_{i=1}^N\mathcal{P}(y_i, f_i)$

---
### Marginal Likelihood (Evidence)

$$
p(y|X, \theta)=\int_f p(y|f,X)\: p(f|X, \theta)\: df
$$



where:
* $p(y|f,X)=\mathcal{N}(y|f, \sigma_n^2\mathbf{I})$
* $p(f|X, \theta)=\mathcal{N}(f|m_\theta, K_\theta)$

Note that all we're doing is simply describing each of these elements specifically because all of these quantities are Gaussian distributed.

$$
p(y|X, \theta)=\int_f \mathcal{N}(y|f, \sigma_n^2\mathbf{I})\cdot \mathcal{N}(f|m_\theta, K_\theta) \: df
$$


So the product of two Gaussians is simply a Gaussian. That along with the notion that the integral of all the functions is a normal distribution with mean $\mu$ and covariance $K$.

$$
p(y|X, \theta)=\mathcal{N}(y|m_\theta, K_\theta + \sigma_n^2 \mathbf{I})
$$


#### Proof:

Using the Gaussian identities:



$$\begin{aligned}
p(x) &= \mathcal{N} (x | \mu, \Lambda^{-1}) \\
p(y|x) &= \mathcal{N} (y | Ax+b, L^{-1}) \\
p(y) &= \mathcal{N} (y|A\mu + b, L^{-1} + A \Lambda^{-1}A^T) \\
p(x|y) &= \mathcal{N} (x|\Sigma \{ A^T L(y-b) + \Lambda\mu \}, \Sigma) \\
\Sigma &= (\Lambda + A^T LA)^{-1}
\end{aligned}$$

So we can use the same reasoning to combine the prior and the likelihood to get the posterior

$$\begin{aligned}
p(f) &= \mathcal{N} (f | m_\theta, \mathbf{K}_\theta) \\
p(y|X) &= \mathcal{N} (y | f(X), \sigma^2\mathbf{I}) \\
p(y) &= \mathcal{N} (y|m_\theta, \sigma_y^2\mathbf{I} + \mathbf{K}_\theta) \\
p(f|y) &= \mathcal{N} (f|\Sigma \{ K^{-1}y + \mathbf{K}_\theta m_\theta \}, \Sigma) \\
\Sigma &= (K^{-1} + \sigma^{-2}\mathbf{I})^{-1}
\end{aligned}$$

**Source**: 

* Alternative Derivation for Log Likelihood - [blog](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html)



---
### Posterior


Alternative Notation:

* $\mathcal{P}(f|y)\propto \mathcal{N}(y|f, \sigma_n^2\mathbf{I})\cdot \mathcal{N}(f|\mu, \mathbf{K}_{ff})$

```python
def posterior(self, X):

    # K(x,x)
    K = self.kernel(self.x_train, self.x_train)

    # K(x,x')
    K_x = self.kernel(self.x_train, X)

    # K(x',x')
    K_xx = self.kernel(X, X)

    # Inverse of kernel
    K_inv = np.linalg.inv(
        K + self.noise_variance * np.eye(len(self.x_train))
    )

    # Calculate the weights
    alpha = K_inv @ self.y_train

    # Calculate the mean function
    mu = K_x @ alpha

    # Calculate the covariance function
    cov = K_xx - K_x.T @ K_inv @ K_x

    return mu, cov
```

---

### Joint Probability Distribution

To make GPs useful, we want to actually make predictions. This stems from the using the joint distribution of the training data and test data with the formula shown above used to condition on multivariate Gaussians. In terms of the GP function space, we have

$$
\begin{aligned}
\mathcal{P}\left(\begin{bmatrix}f \\ f_*\end{bmatrix} \right) &= 

 \mathcal{N}\left( 
    \begin{bmatrix}
    \mu \\ \mu_*
    \end{bmatrix},
    \begin{bmatrix}
    K_{xx} & K_{x*} \\ K_{*x} & K_{**}
    \end{bmatrix} \right)
    \end{aligned}
$$


Then solving for the marginals, we can come up with the predictive test points.

$$\mathcal{P}(f_* |X_*, y, X, \theta)= \mathcal{N}(f_* | \mu_*, \nu^2_*  )$$

where:

* $\mu*=K_* (K + \sigma^2 I)^{-1}y=K_* \alpha$
* $\nu^2_*= K_{**} - K_*(K + \sigma^2I)^{-1}K_*^{\top}$

---
### Learning in GPs

The prior $m(x), K$ have hyper-parameters $\theta$. So learning a $\mathcal{GP}$ implies inferring hyper-parameters from the model. 

$$\mathcal{P}(Y|X,\theta)=\int \mathcal{P}(Y|f)\mathcal{P}(f|X, \theta)df$$

However, we are not interested in $f$ directly. We can marginalize it out via the integral equation. The marginal of a Gaussian is Gaussian.

**Note**: Typically we use the $\log$ likelihood instead of a pure likelihood. This is purely for computational purposes. The $\log$ function is monotonic so it doesn't alter the location of the extreme points of the function. Furthermore we typically minimize the $-\log$ instead of the maximum $\log$ for purely practical reasons.

One way to train these functions is to use Maximum A Posterior (MAP) of the hyper-parameters


$$
\begin{aligned}
\theta^* &= \underset{\theta}{\text{argmax}}\log p(y|X,\theta) \\
&= \underset{\theta}{\text{argmax}}\log \mathcal{N}(y | 0, K + \sigma^2 I)
\end{aligned}
$$


### Maximum Likelihood

$$
\log p(y|x, \theta) = - \frac{N}{2} \log 2\pi - \frac{1}{2}y^{\top}(K+\sigma^2I)^{-1}y - \frac{1}{2} \log \left| K+\sigma^2I \right| 
$$

In terms of the cholesky decomposition:

Let $\mathbf{L}=\text{cholesky}(\mathbf{K}+\sigma_n^2\mathbf{I})$. We can write the log likelihood in terms of the cholesky decomposition.
$$
\begin{aligned}
\log p(y|x, \theta) &= - \frac{N}{2} \log 2\pi - \frac{1}{2}y^{\top}(K+\sigma^2I)^{-1}y - \frac{1}{2} \log \left| K+\sigma^2I \right|  \\
&= - \frac{N}{2} \log 2\pi - \frac{1}{2} ||\mathbf{L}^{-1}y||^2 - \sum_i \log \mathbf{L}_{ii} 
\end{aligned}
$$


This gives us a computational complexity of $\mathcal{O}(N + N^2 + N^3)=\mathcal{O}(N^3)$

```python
from scipy.linalg.lapack import dtrtrs
from scipy.linalg import cholesky, cho_solve

def log_likelihood(self, X, y):
    n_samples = self.x_train.shape[0]

    K = self.kernel(X)

    K_gp = K + self.noise_variance * np.eye(n_samples)

    L = np.linalg.cholesky(K_gp)

    LinvY =  dtrtrs(L, y, lower=1)[0]
	
  	# term I - constant
    logL = - (n_samples / 2.0) * np.log(2 * np.pi)
	
  	# term II - inverse
    logL += - (1.0 / 2.0) * np.square(LinvY).sum()
		
    # term III - determinant
    logL += - np.log(np.diag(L)).sum()

    return logL
```

source - Dai, [GPSS 2018](http://zhenwendai.github.io/slides/gpss2018_slides.pdf)



### Minimizing the log-likelihood function:

#### Term I

$$
\frac{\partial}{\partial \theta} \left( K + \sigma^2I  \right)^{-1}=- (K+\sigma^2I)^{-1} \frac{\partial}{\partial \theta} \left( K + \sigma^2I  \right)(K+\sigma^2I)^{-1}
$$

#### Term II

$$
\begin{aligned}
\frac{\partial}{\partial \theta} \log \left| K + \sigma^2I \right|
&=\text{trace }\left( \frac{\partial}{\partial \theta} \log (K + \sigma^2 I) \right) \\ 
&=\text{trace }\left( (K+\sigma^2I)^{-1} \frac{\partial}{\partial \theta}  (K + \sigma^2 I) \right)
\end{aligned}
$$



**Rule:** $\log |\text{det }A|=\text{trace }(\log A)$

---

[**Learning with GPs**](https://www.youtube.com/watch?v=Z9cdlQ-WDLM&index=7&list=PLAbhVprf4VPlqc8IoCi7Qk0YQ5cPQz9fn)

---
## Resources

#### Formulation
* [Multivariate Gaussian Distribution](https://peterroelants.github.io/posts/multivariate-normal-primer/)
* Zhoubin - [GP Talk I](https://www.youtube.com/watch?v=IEpc2ClaYH8&list=PLAbhVprf4VPlqc8IoCi7Qk0YQ5cPQz9fn&index=6)
* Maurizio Filippone - [GPS](https://www.youtube.com/watch?v=4-pvFVd_eEQ)
* [Survey of GPs for EO](https://www.uv.es/lapeva/papers/2016_IEEE_GRSM.pdf)


#### Visualization

* [Understanding GPs](https://peterroelants.github.io/posts/gaussian-process-tutorial/) | [Fitting GP Kernel](https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/) | [GP Kernels](https://peterroelants.github.io/posts/gaussian-process-kernels/)
* [A Visual Exploration of GPs](https://distill.pub/2019/visual-exploration-gaussian-processes/)
* [A Practical Guide to GPs](https://drafts.distill.pub/gp/)
* [D3 GPR Demo](http://www.tmpl.fi/gp/)

#### Code Tutorials
* [GPs](http://krasserm.github.io/2018/03/19/gaussian-processes/)
  
  > A really good coding tutorial about how to implement GPs. Also compares how different libraries implement GPs (sklearn, GPy)
* [Understanding GPs](https://peterroelants.github.io/posts/gaussian-process-tutorial/)
* [Fitting GP Models in Python](https://blog.dominodatalab.com/fitting-gaussian-process-models-python/)
* [GP Lecture Notebook](https://nbviewer.jupyter.org/github/adamian/adamian.github.io/blob/master/talks/Brown2016.ipynb)
* [Using CuPy](https://github.com/ericmjl/bayesian-analysis-recipes/blob/master/notebooks/gp-cupy.ipynb)

#### GP Kernels
* [The Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)

#### Simple Complete Implementations

* [NeuGap](https://github.com/Alaya-in-Matrix/NeuGaP/blob/master/Model.py)



---
---
## Supplementary



### Tips and Tricks for Practicioners

* Set initial hyper-parameters with domain knowledge
* Standardize input data
  * Set initial length scales to $\sigma_l \approx 0.5$
* Standardize targets $\mathbf{y}$
  * Set initial signal variance to $\sigma_f\approx 1.0$
* Set noise level initially high $\sigma_n \approx 0.5 \times \sigma_f$
* Random restarts
* Penalize high signal-to-noise ratios ($\frac{\sigma_f}{\sigma_n}$)

**Source**: 
* Practical Guide to GPs - [Interactive Blog](https://drafts.distill.pub/gp/)
* Deisenroth - [Prezi](https://drive.google.com/file/d/1i9D7PTgc4KKT-nYQ2qxzbJNcOuCyu0d5/view)


### Limitations of GPs

The primary problem is the computational and memory complexity

With a training size $N$ and $D$ dimensions, here are the complexity estimates for all parts of the GP algorithm:
* Training: $\mathcal{O}(N^3)$
* Predictive variance: $\mathcal{O}(N^2)$
* Memory: $\mathcal{O}(ND + N^2)$

So this gives us a practical limit of about $N \approx 10K$ training points. There are many solutions to this problem but with the exact GP formulation without any clever tricks, you should be aware of this.

---
### Plan

Implement a GP from scratch and also do it in an OOP framework. This will give me an opportunity to go through the motions of thinking in an OOP manner as well as introduce some key points to a GP. Probably it would have the following format:

1. Start with the fundamentals of a GP; data, sampling, training, etc
2. Go through and OOP-fy it. Initializations, kernels, etc.



---
#### Books


#### Important Papers

---
#### Thesis Explain

Often times the papers that people publish in conferences in Journals don't have enough information in them. Sometimes it's really difficult to go through some of the mathematics that people put  in their articles especially with cryptic explanations like "it's easy to show that..." or "trivially it can be shown that...". For most of us it's not easy nor is it trivial. So I've included a few thesis that help to explain some of the finer details. I've arranged them in order starting from the easiest to the most difficult.


* [GPR Techniques](https://github.com/HildoBijl/GPRT) - Bijl (2016)    
  * Chapter V - Noisy Input GPR
* [Efficient Reinforcement Learning Using Gaussian Processes](https://www.semanticscholar.org/paper/Efficient-reinforcement-learning-using-Gaussian-Deisenroth/edab384ff0d582807b7b819bcc79eff8cda8a0ef) - Deisenroth (2009)