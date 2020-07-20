# Resources

## Pairwise Distances



## Libraries

### Gaussian Processes

* [LADAX](https://github.com/danieljtait/ladax)
  > GPs and DeepGPs. Using Layers of distributions.

```python
class SVGP(nn.Module):
    def apply(self, x):
        kernel_fn = kernel_provider(x, **kernel_fn_kwargs)
        inducing_var = inducing_variable_provider(x, kernel_fn, **inducing_var_kwargs)
        vgp = SVGPLayer(x, mean_fn, kernel_fn, inducing_var)
        return vgp
```
### MonteCarlo

* Numpyro
* MCX




### Kernel Methods

* Kernel Density Estimation - [jax_cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/redshift.py)
* [Gaussian Kernel](https://github.com/JohnYKiyo/density_ratio_estimation/blob/master/src/densityratio/densityratio.py#L171)

### Polynomial

* Interpolation - [jax_cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/bc360028e5ff92160e725388f26af39457a8d068/jax_cosmo/scipy/interpolate.py)


### Normalizing Flows

* Normalizing Flows in JAX - [ChrisWaites](https://github.com/ChrisWaites/jax-flows)


### Statistical Learning

* Rethinking Statistical Learning - [Fehlepsi](https://github.com/fehiepsi/rethinking-numpyro)
