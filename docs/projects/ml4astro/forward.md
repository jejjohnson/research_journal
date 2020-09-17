# Physics-Based Model


## Ideas

1. Refactor Physics Model
2. Get Toy Datasets
3. Implement Different samplers
4. Do generic visualizations

---

## Samplers

We want to work with some of the generic samplers as well as some of the SOTA methods. We also want to use things out of the box and we don't want to reinvent algorithms.

* Affine-Variant Sampling Ensemble Sampler - `emcee`
* Nested Sampling - `dynesty`
* NUTS - `pymc3`
* SMC - `pymc3`

We can also integrate some of this code based on some code by Dan! We can see an example [here](https://dfm.io/posts/emcee-pymc3/) where we integrate `emcee` with `pymc3`. If the formatting is the same, we should also be able to use the nested sampler from `dynesty`.

---

## Visualization

We want to do generic visualizations. Meaning we should be able to use the same functions for other models. And it will also be using a common platform. In this case, it will be `arviz`. 