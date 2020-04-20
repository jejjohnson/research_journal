---
title: Basic GP
description: A Basic GP Built from scratch
authors:
    - J. Emmanuel Johnson
path: docs/appendix/gps
source: 1_introduction.md
---
# GP from Scratch

This post will go through how we can build a GP regression model from scratch. I will be going over the formulation as well as how we can code this up from scratch. I did this before a long time ago but I've learned a lot about GPs since then. So I'm putting all of my knowledge together so that I can get a good implementation that goes in parallel with the theory. I am also interested in furthering my research on [uncertain GPs](../../projects/ErrorGPs/README.md) where I go over how we can look at input error in GPs.


!!! info "Materials"
    The full code can be found in the colab notebook. Later I will refactor everything into a script so I can use it in the future.

    * [Colab Notebook](https://colab.research.google.com/drive/1JQy7nsNOmkfDm_ovCQQ0zUtx2hwAI4ll)

!!! check "Good News"
    It took me approximately 12 hours in total to code this up from scratch. That's significantly better than last time as that time easily took me a week and some change. And I still had problems with the code afterwards. That's progress, no?

!!! info "Resources"
    I saw quite a few tutorials that inspired me to do this tutorial.

    * [Blog Post](http://krasserm.github.io/2018/03/19/gaussian-processes/) - 
    > Excellent blog post that goes over GPs with step-by-step. Necessary equations only.
    * [Blog Post Series](https://peterroelants.github.io/posts/gaussian-process-tutorial/) - Peter Roelants
    > Good blog post series that go through more finer details of GPs using TensorFlow.

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

### Bayesian Treatment

So now how does this look in terms of the Bayes theorem in words:

$$
\text{Posterior} = \frac{\text{Likelihood}\cdot\text{Prior}}{\text{Evidence}}
$$

And mathematically:

$$
p(f|X,y) = \frac{p(y|f, X) \: p(f|X, \theta)}{p(y| X)}
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




#### Sampling from Prior

Now, something a bit more practical, generally speaking when we program the sampling portion of the prior, we need data. The kernel function is as is and has already been defined with its appropriate parameters. Furthermore, we already have defined the mean function $\mu$ when we initialized the mean function above. So we just need to pass the function through the multivariate normal function along with the number of samples we would like to draw from the prior.

!!! details "Code"
    ```python
    # initialize parameters
    params = {
        'gamma': 10., 
        'length_scale': 1e-3, 
    }

    n_samples = 10                   # condition on 10 samples 
    test_X = X[:n_samples, :].copy() # random samples from data distribution

    # GP Prior functions (mu, sigma)
    mu_f = zero_mean                            
    cov_f = functools.partial(gram, rbf_kernel)
    mu_x, cov_x = gp_prior(params, mu_f=mu_f, cov_f=cov_f , x=test_X)

    # make it semi-positive definite with jitter
    jitter = 1e-6
    cov_x_ = cov_x + jitter * jnp.eye(cov_x.shape[0])

    n_functions = 10                # number of random functions to draw

    key = jax.random.PRNGKey(0)     # Jax random numbers boilerplate code

    y_samples = jax.random.multivariate_normal(key, mu_x, cov_x_, shape=(n_functions,))
    ```

---

## Likelihood (noise model)

$$
p(y|f,X)=\prod_{i=1}^{N}\mathcal{N}(y_i|f_i,\sigma_\epsilon^2)= \mathcal{N}(y|f(x), \sigma_\epsilon^2\mathbf{I}_N)
$$



This comes from our assumption as stated above from $y=f(x)+\epsilon$.

Alternative Notation:
* $y\sim \mathcal{N}(f, \sigma_n^2)$
* $\mathcal{N}(f, \sigma_n^2) = \prod_{i=1}^N\mathcal{P}(y_i, f_i)$

---

## Posterior


Alternative Notation:

* $\mathcal{P}(f|y)\propto \mathcal{N}(y|f, \sigma_n^2\mathbf{I})\cdot \mathcal{N}(f|\mu, \mathbf{K}_{ff})$


!!! details "Code"

    This will easily be the longest function that we need for the GP. In my version, it's not necessary for training the GP. But it is necessary for testing.


    === "Posterior"

        ```python
        def posterior(params, prior_params, X, Y, X_new, likelihood_noise=False, return_cov=False):
            (mu_func, cov_func) = prior_params

            # ==========================
            # 1. GP PRIOR
            # ==========================
            mu_x, Kxx = gp_prior(params, mu_f=mu_func, cov_f=cov_func, x=X)

            # ===========================
            # 2. CHOLESKY FACTORIZATION
            # ===========================

            (L, lower), alpha = cholesky_factorization(
                Kxx + (params["likelihood_noise"] + 1e-7) * jnp.eye(Kxx.shape[0]), 
                Y-mu_func(X).reshape(-1,1)
            )

            # ================================
            # 4. PREDICTIVE MEAN DISTRIBUTION
            # ================================

            # calculate transform kernel
            KxX = cov_func(params, X_new, X)

            # Calculate the Mean
            mu_y = jnp.dot(KxX, alpha)

            # =====================================
            # 5. PREDICTIVE COVARIANCE DISTRIBUTION
            # =====================================
            v = jax.scipy.linalg.cho_solve((L, lower), KxX.T)
            
            # Calculate kernel matrix for inputs
            Kxx = cov_func(params, X_new, X_new)
            
            cov_y = Kxx - jnp.dot(KxX, v)

            # Likelihood Noise
            if likelihood_noise is True:
                cov_y += params['likelihood_noise']

            # return variance (diagonals of covariance)
            if return_cov is not True:
                cov_y = jnp.diag(cov_y)

            return mu_y, cov_y
        ```

!!! info "Cholesky"

    A lot of times just straight solving the $K^{-1}y=\alpha$ will give you problems. Many times you'll get an error about the matrix being ill-conditioned and non positive semi-definite. So we have to rectify that with the Cholesky decomposition. $K$ should be a positive semi-definite matrix so, there are more stable ways to solve this. We can use the cholesky decomposition which decomposes $K$ into a product of two lower triangular matrices:

    $$K = LL^\top$$

    We do this because:

    1. it's less expensive to calculate the inverse of a triangular matrix
    2. it's easier to solve systems of equations $Ax=b$.


    === "Cholesky Factorization"

        There are two convenience terms that allow you to calculate the cholesky decomposition:

        1. `cho_factor` - calculates the decomposition $K \rightarrow L$
        2. `cho_solve` - solves the system of equations problem $LL^\top \alpha=y$

        ```python
        def cholesky_factorization(K, Y):

            # cho factor the cholesky, K = LL^T
            L = jax.scipy.linalg.cho_factor(K, lower=True)

            # alpha, LL^T alpha=y
            alpha = jax.scipy.linalg.cho_solve(L, Y)

            return L, alpha
        ```

        **Note**: If you want to get the cholesky matrix by itself and operator on it without the `cho_factor` function, then you should call the `cholesky` function directly. The `cho_factor` puts random (inexpensive) values in the part of the triangle that's not necessary. Whereas the `cholesky` adds zeros there instead.

    === "Variance Term"

        The variance term also makes use of the $K^{-1}$. So naturally, we can use the already factored cholesky decompsition to calculate the term.

        $$
        \begin{aligned}
        k_* K^{-1}k_*
        &= (Lv)^\top K^{-1}Lv\\
        &= v^\top L^\top (LL^\top)^{-1} Lv\\
        &= v^{\top}L^\top L^{-\top}L^{-1}v\\
        &= v^\top v
        \end{aligned}
        $$

        ```python

        v = jax.scipy.linalg.cho_solve((L, lower), KxX.T)
        var = np.dot(KxX, v)
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

## Marginal Log-Likelihood


The prior $m(x), K$ have hyper-parameters $\theta$. So learning a $\mathcal{GP}$ implies inferring hyper-parameters from the model. 

$$p(Y|X,\theta)=\int p(Y|f)p(f|X, \theta)df$$

However, we are not interested in $f$ directly. We can marginalize it out via the integral equation. The marginal of a Gaussian is Gaussian.

**Note**: Typically we use the $\log$ likelihood instead of a pure likelihood. This is purely for computational purposes. The $\log$ function is monotonic so it doesn't alter the location of the extreme points of the function. Furthermore we typically minimize the $-\log$ instead of the maximum $\log$ for purely practical reasons.

One way to train these functions is to use Maximum A Posterior (MAP) of the hyper-parameters


$$
\begin{aligned}
\theta^* &= \underset{\theta}{\text{argmax}}\log p(y|X,\theta) \\
&= \underset{\theta}{\text{argmax}}\log \mathcal{N}(y | 0, K + \sigma^2 I)
\end{aligned}
$$


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

---

### Marginal Log-Likelihood

!!! todo "TODO"
    Proof of Marginal Log-Likelihood

Now we need a cost function that will allow us to get the best hyperparameters that fit our data.

$$
\log p(y|x, \theta) = - \frac{N}{2} \log 2\pi - \frac{1}{2}y^{\top}(K+\sigma^2I)^{-1}y - \frac{1}{2} \log \left| K+\sigma^2I \right| 
$$

Inverting $N\times N$ matrices is the worse part about GPs in general. There are many techniques to be able to handle them, but for basics, it can become a problem. Furthermore, inverting this Kernel matrix tends to have problems being *positive semi-definite*. One way we can make this more efficient is to do the cholesky decomposition and then solve our problem that way. 

#### Cholesky Components

Let $\mathbf{L}=\text{cholesky}(\mathbf{K}+\sigma_n^2\mathbf{I})$. We can write the log likelihood in terms of the cholesky decomposition.

$$
\begin{aligned}
\log p(y|x, \theta) &= - \frac{N}{2} \log 2\pi - \frac{1}{2} ||\underbrace{\mathbf{L}^{-1}y}_{\alpha}||^2 - \sum_i \log \mathbf{L}_{ii}
\end{aligned}
$$


This gives us a computational complexity of $\mathcal{O}(N + N^2 + N^3)=\mathcal{O}(N^3)$


!!! details "Code"

    I will demonstrate two ways to do this: 

    1. We will use the equations above
    2. We will refactor this and use the built-in function

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
            (L, lower), alpha = cholesky_factorization(
                Kxx + ( params['likelihood_noise'] + 1e-5 ) * jnp.eye(Kxx.shape[0]), Y
            )

            # ===========================
            # 3. Marginal Log-Likelihood
            # ===========================
            log_likelihood = -0.5 * jnp.einsum("ik,ik->k", Y, alpha) # same as dot(Y.T, alpha)
            log_likelihood -= jnp.sum(jnp.log(jnp.diag(L)))
            log_likelihood -= ( Kxx.shape[0] / 2 ) * jnp.log(2 * jnp.pi)

            return - log_likelihood.sum()
        ```

    === "Refactored"
        ```python
        def marginal_likelihood(prior_params, params,  Xtrain, Ytrain):
            
            # unpack params
            (mu_func, cov_func) = prior_params
            
            # ==========================
            # 1. GP Prior, mu(), cov(,)
            # ==========================
            mu_x = mu_f(Ytrain)
            Kxx = cov_f(params, Xtrain, Xtrain)
            
            # ===========================
            # 2. GP Likelihood
            # ===========================
            K_gp = Kxx + ( params['likelihood_noise'] + 1e-6 ) * jnp.eye(Kxx.shape[0])
            
            # ===========================
            # 3. Marginal Log-Likelihood
            # ===========================
            # get log probability
            log_prob = jax.scipy.stats.multivariate_normal.logpdf(x=Ytrain.T, mean=mu_x, cov=K_gp)

            # sum dimensions and return neg mll
            return -log_prob.sum()
        ```

source - Dai, [GPSS 2018](http://zhenwendai.github.io/slides/gpss2018_slides.pdf)




## Training

!!! details "Code"

    === "Log Params"

        We often have problems when it comes to using optimizers. A lot of times they just don't seem to want to converge and the gradients seem to not change no matter what happens. One trick we can do is to make the optimizer solve a transformed version of the parameters. And then we can take a softmax so that they converge properly.

        $$f(x) = \ln (1 + \exp(x))$$

        Jax has a built-in function so we'll just use that.

        ```python
        def saturate(params):
            return {ikey:jax.nn.softplus(ivalue) for (ikey, ivalue) in params.items()}
        ```

    === "Experimental Parameters"

        ```python
        logger.setLevel(logging.INFO)

        X, y, Xtest, ytest = get_data(50)


        # PRIOR FUNCTIONS (mean, covariance)
        mu_f = zero_mean
        cov_f = functools.partial(gram, rbf_kernel)
        gp_priors = (mu_f, cov_f)

        # Kernel, Likelihood parameters
        params = {
            'gamma': 2.0,
            # 'length_scale': 1.0,
            # 'var_f': 1.0,
            'likelihood_noise': 1.,
        }
        # saturate parameters with likelihoods
        params = saturate(params)

        # LOSS FUNCTION
        mll_loss = jax.jit(functools.partial(marginal_likelihood, gp_priors))

        # GRADIENT LOSS FUNCTION
        dloss = jax.jit(jax.grad(mll_loss))
        ```

    === "Training Step"

        ```python
        # STEP FUNCTION
        @jax.jit
        def step(params, X, y, opt_state):
            # calculate loss
            loss = mll_loss(params, X, y)

            # calculate gradient of loss
            grads = dloss(params, X, y)

            # update optimizer state
            opt_state = opt_update(0, grads, opt_state)

            # update params
            params = get_params(opt_state)

            return params, opt_state, loss
        ```
    
    === "Experimental Loop"

        ```python
        # initialize optimizer
        opt_init, opt_update, get_params = optimizers.rmsprop(step_size=1e-2)

        # initialize parameters
        opt_state = opt_init(params)

        # get initial parameters
        params = get_params(opt_state)

        # TRAINING PARARMETERS
        n_epochs = 500
        learning_rate = 0.1
        losses = list()
        postfix = {}

        import tqdm

        with tqdm.trange(n_epochs) as bar:

            for i in bar:
                # 1 step - optimize function
                params, opt_state, value = step(params, X, y, opt_state)

                # update params
                postfix = {}
                for ikey in params.keys():
                    postfix[ikey] = f"{jax.nn.softplus(params[ikey]):.2f}"

                # save loss values
                losses.append(value.mean())

                # update progress bar
                postfix["Loss"] = f"{onp.array(losses[-1]):.2f}"
                bar.set_postfix(postfix)
                # saturate params
                params = saturate(params)
        ```

---

## Resources

* Surrogates: GP Modeling, Design, and Optimization for the Applied Sciences - Gramacy - [Online Book](https://bookdown.org/rbg/surrogates/)