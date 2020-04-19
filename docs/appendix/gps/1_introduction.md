---
title: Basic GP
description: A Basic GP Built from scratch
authors:
    - J. Emmanuel Johnson
path: docs/appendix/gps
source: 1_introduction.md
---
# GP from Scratch


## Definition

The first thing to understand about GPs is that we are actively placing a distribution $\mathcal{P}(f)$ on functions $f$ where these functions can be infinitely long function values $f=[f_1, f_2, \ldots]$. A GP generalizes the multivariate Gaussian distribution to infinitely many variables.

> A GP is a collection of random variables $f_1, f_2, \ldots$, any finite number of which is Gaussian distributed.

> A GP defines a distribution over functions $p(f)$ which can be used for Bayesian regression. (Zhoubin)

Another nice definition is:

> **Gaussian Process**: Any set of function variables $\{f_n \}^{N}_{n=1}$ has a joint Gaussian distribution with mean function $m$. (Deisenroth)

The nice thing is that this is provided by a mean function $\mu$ and covariance matrix $\mathbf{K}$

---

## Bayesian Inference Problem

#### Objective


Let's have some data set, $\mathcal{D}= \left\{ (x_i, y_i)^N_{i=1} \right\}=(X,y)$

#### Model

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

#### Bayesian Treatment

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


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
class GPR:
    def __init__(self, mu, kernel, X, y, noise_variance=1e-6):
        self.mu = mu
        self.kernel = kernel
        self.x_train = x_train
        self.y_train = y_train
        self.noise_variance = noise_variance
```
</details>

---

## Gaussian Process Prior

This is the basis of the GP method. Under the assumption that we mentioned above:

$$
p(f|X, \theta)=\mathcal{GP}(m_\theta , \mathbf{K}_\theta)
$$

where $m_\theta$ is a mean function and $\mathbf{K}$ is a covariance function

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
\end{aligned}
$$

and we arrive at a joint Gaussian distribution of the training and testing which is still normally distributed due to the marginalization.

#### Code

!!! example "Code"
    === "Mean Function"

        Honestly, I never work with mean functions. I always assume a zero-mean function and that's it. I don't really know anyone who works with mean functions either. I've seen it used in deep Gaussian processes but I have no expertise in which mean functions to use. So, we'll follow the community standard for now: zero mean function.
        ```python
        def zero_mean(x):
            return jnp.zeros(x.shape[0])
        ```

        The output of the mean function is size $\mathbb{R}^{N}$.

    === "Kernel Function"

        The most common kernel function you will see in the literature is the Radial Basis Function (RBF). It's a universal approximator and it performs fairly well on **most** datasets. If your dataset becomes non-linear, then it may start to fail as it is a really smooth function. The kernel function is defined as:

        $$
        k(x,y) = \sigma_f \exp \left( - \gamma || x - y||^2_2 \right)
        $$

        ```python

        # Squared Euclidean Distance Formula
        @jax.jit
        def sqeuclidean_distance(x, y):
            return jnp.sum((x-y)**2)

        # RBF Kernel
        @jax.jit
        def rbf_kernel(params, x, y):
            return jnp.exp( - params['gamma'] * sqeuclidean_distance(x, y))
        ```

        We also have the more robust version of the RBF with a separate length scale per dimension called the Automatic Relavance Determination (ARD) kernel.

        $$
        k(x,y) = \sigma_f \exp \left( - || x / \sigma_\lambda - y / \sigma_\lambda ||^2_2 \right)
        $$

        ```python
        # ARD Kernel
        @jax.jit
        def ard_kernel(params, x, y):
            
            # divide by the length scale
            x = x / params['length_scale']
            y = y / params['length_scale']
            
            # return the ard kernel
            return params['var_f'] * jnp.exp( - sqeuclidean_distance(x, y) )
        ```
        **Remember**: These are functions so they take in vectors $\mathbf{x} \in  \mathbb{R}^{D}$ and output a scalar value.

    === "Kernel Matrix"

        The kernel function in the tab over shows how we can calculate the kernel for an input vector. But we need every single combination
    

        ```python
        # Gram Matrix
        def gram(func, params, x, y):
            return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)
        ```





Now, something a bit more practical, generally speaking when we program the sampling portion of the prior, we need data. The kernel function is as is and has already been defined with its appropriate parameters. Furthermore, we already have defined the mean function $\mu$ when we initialized the mean function above. So we just need to pass the function through the multivariate normal function along with the number of samples we would like to draw from the prior.

<details>

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

</details>

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


??? details "Proof"

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



## Posterior


Alternative Notation:

* $\mathcal{P}(f|y)\propto \mathcal{N}(y|f, \sigma_n^2\mathbf{I})\cdot \mathcal{N}(f|\mu, \mathbf{K}_{ff})$


??? details "Code"

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


## Marginal Log-Likelihood

$$
\log p(y|x, \theta) = - \frac{N}{2} \log 2\pi - \frac{1}{2}y^{\top}(K+\sigma^2I)^{-1}y - \frac{1}{2} \log \left| K+\sigma^2I \right| 
$$

In terms of the cholesky decomposition:

Let $\mathbf{L}=\text{cholesky}(\mathbf{K}+\sigma_n^2\mathbf{I})$. We can write the log likelihood in terms of the cholesky decomposition.

$$
\begin{aligned}
\log p(y|x, \theta) &= - \frac{N}{2} \log 2\pi - \frac{1}{2}y^{\top}(K+\sigma^2I)^{-1}y - \frac{1}{2} \log \left| K+\sigma^2I \right|  \\
&= - \frac{N}{2} \log 2\pi - \frac{1}{2} ||\mathbf{L}^{-1}y||^2 - \sum_i \log \mathbf{L}_{ii} \\
&= - \frac{N}{2} \log 2\pi - \frac{1}{2} ||\alpha||^2 - \sum_i \log \mathbf{L}_{ii}
\end{aligned}
$$


This gives us a computational complexity of $\mathcal{O}(N + N^2 + N^3)=\mathcal{O}(N^3)$


!!! details "Code"

    === "From Scratch"
        ```python
        def nll_scratch(gp_priors, params, X, Y) -> float:
            
            (mu_func, cov_func) = gp_priors
            
            # ==========================
            # 1. GP PRIOR
            # ==========================
            mu_x, Kxx = gp_prior(params, mu_f=mu_func, cov_f=cov_func , x=X)
            
            # ===========================
            # 2. CHOLESKY FACTORIZATION
            # ===========================
            (L, lower), alpha = cholesky_factorization(Kxx + ( params['likelihood_noise'] + 1e-5 ) * jnp.eye(Kxx.shape[0]), Y)

            # ===========================
            # 3. Marginal Log-Likelihood
            # ===========================
            log_likelihood = -0.5 * jnp.einsum("ik,ik->k", Y, alpha)
            log_likelihood -= jnp.sum(jnp.log(jnp.diag(L)))
            log_likelihood -= ( Kxx.shape[0] / 2 ) * jnp.log(2 * jnp.pi)

            return - jnp.sum(log_likelihood)
        ```

    === "Refactored"
        ```python
        def marginal_likelihood(prior_params, params,  Xtrain, Ytrain):
            
            # unpack params
            (mu_func, cov_func) = prior_params
            
            # ==========================
            # 1. GP Prior (mu, sig)
            # ==========================
            mu_x = mu_f(Xtrain)
            Kxx = cov_f(params, Xtrain, Xtrain)
            
            # ===========================
            # 2. GP Kernel
            # ===========================
            K_gp = Kxx + ( params['likelihood_noise'] + 1e-6 ) * jnp.eye(Kxx.shape[0])
            
            # ===========================
            # 3. Built-in GP Likelihood
            # ===========================
            log_prob = jax.scipy.stats.multivariate_normal.logpdf(Ytrain.squeeze(), mean=jnp.zeros(Ytrain.shape[0]), cov=K_gp)
            
            nll = jnp.sum(log_prob)
            return -nll
        ```

source - Dai, [GPSS 2018](http://zhenwendai.github.io/slides/gpss2018_slides.pdf)




## Training

!!! details "Code"

    === "Train Step"

    ```python
    logger.setLevel(logging.INFO)

    X, y, Xtest, ytest = get_data(30)



    params = {
        'gamma': 10.,
    #     'length_scale': 1.0,
    #     'var_f': 1.0,
        'likelihood_noise': 1e-3,
    }

    # Nice Trick for better training of params
    def saturate(params):
        return {ikey:softplus(ivalue) for (ikey, ivalue) in params.items()}

    params = saturate(params)

    cov_f = functools.partial(gram, rbf_kernel)

    gp_priors = (mu_f, cov_f)

    # LOSS FUNCTION
    mll_loss = jax.jit(functools.partial(nll_scratch, gp_priors))

    # GRADIENT LOSS FUNCTION
    dloss = jax.jit(jax.grad(mll_loss))



    # MEAN FUNCTION
    mu_f = zero_mean


    # l_val = mll_loss(saturate(params), X[0,:], y[0, :].reshape(-1, 1))
    l_vals = mll_loss(saturate(params), X, y)
    # print('MLL (vector):', l_val)
    # print('MLL (samples):', l_vals)


    # dl_val = dloss(saturate(params), X[0,:], y[0, :].reshape(-1, 1))
    dl_vals = dloss(saturate(params), X, y)
    # print('dMLL (vector):', dl_val)|
    # print('dMLL (samples):', dl_vals)



    # STEP FUNCTION
    @jax.jit
    def step(params, X, y, opt_state):
        # print("BEOFRE!")
        # print(X.shape, y.shape)
        # print("PARAMS", params)
        # print(opt_state)
        # value and gradient of loss function
        loss = mll_loss(params, X, y)
        grads = dloss(params, X, y)
        # # print(f"VALUE:", value)
        # print("During! v", value)
        # print("During! p", params)
        # print("During! g", grads)
        # update parameter state
        opt_state = opt_update(0, grads, opt_state)

        # get new params
        params = get_params(opt_state)
        # print("AFTER! v", value)
        # print("AFTER! p", params)
        # print("AFTER! g", grads)
        return params, opt_state, loss

    # initialize optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)

    # initialize parameters
    opt_state = opt_init(params)

    # get initial parameters
    params = get_params(opt_state)
    # print("PARAMS!", params)

    n_epochs = 2_000
    learning_rate = 0.01
    losses = list()

    import tqdm

    with tqdm.trange(n_epochs) as bar:

        for i in bar:
            postfix = {}
    #         params = saturate(params)
            # get nll and grads
            # nll, grads = dloss(params, X, y)

            params, opt_state, value = step(params, X, y, opt_state)

            # update params
            # params, momentums, scales, nll = train_step(params, momentums, scales, X, y)
            for ikey in params.keys():
                postfix[ikey] = f"{params[ikey]:.2f}"
            # params[ikey] += learning_rate * grads[ikey].mean()

            losses.append(value.mean())
            postfix["Loss"] = f"{onp.array(losses[-1]):.2f}"
            bar.set_postfix(postfix)
            params = saturate(params)
            

    # params = log_params(params)
    ```

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