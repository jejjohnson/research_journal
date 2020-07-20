# MonteCarlo

---

## [Numpyro](https://github.com/pyro-ppl/numpyro)

> Probabilistic programming with numpy by Jax for autograd and JIT compilation to GPU/TPU/CPU.

### Predict

```python
def predict(model, rng_key, samples, X):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    Y = model_trace["Y"]["value"]
    return Y
```

### Sample

```python
def sample(
    model,
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    seed: int,
    chain_method: str="parallel",
    summary: bool=True,
    **kwargs: Dict={},
):
    # generate random key
    rng_key = random.PRNGKey(seed)
    # generate model from NUTS
    kernel = NUTS(model)
    # Note: sampling
    mcmc = MCMC(kernel, n_warmup, n_samples, n_chains, chain_method=chain_method)
    mcmc.run(rng_key, **kwargs)

    if summary:
        mcmc.print_summary()
    return mcmc
```

---

## [MCX](https://github.com/rlouf/mcx) 

> A library to compile probabilitistc programs for performant Inference on CPU & GPU

```python
from jax import numpy as np
import mcx
import mcx.distributions as dist

x_data = np.array([2.3, 8.2, 1.8])
y_data = np.array([1.7, 7., 3.1])

@mcx.model
def linear_regression(x, lmbda=1.):
    scale @ dist.Exponential(lmbda)
    coefs @ dist.Normal(np.zeros(np.shape(x)[-1]))
    y = np.dot(x, coefs)
    predictions @ dist.Normal(y, scale)
    return predictions

rng_key = jax.random.PRNGKey(0)

# Sample the model forward, conditioning on the value of `x`
mcx.sample_forward(
    rng_key,
    linear_regression,
    x=x_data,
    num_samples=10_000
)

# Sample from the posterior distribution using HMC
kernel = mcx.HMC(
    step_size=0.01,
    num_integration_steps=100,
    inverse_mass_matrix=np.array([1., 1.]),
)

observations = {'x': x_data, 'predictions': y_data, 'lmbda': 3.}
sampler = mcx.sample(
    rng_key,
    linear_regression,
    kernel,
    **observations
)
trace = sampler.run()
```


