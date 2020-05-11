# Bisection search

This is a very interesting thing that you can do instead of interpolation. This comes from the fact that **PyTorch** doesn't have a native interpolation algorithm. Although there are saints on the internet who have implemented their own native algorithm, the library itself doesn't have a core one. I find it very interesting that they don't and I'm not sure why. Is there something difficult about automatic differentiation and interpolation? 

In any case, one thing that I did see was bisection search instead of interpolation. I've also seen it in the `rv_histogram` function in the scipy library. This function allows you to construct an empirical distribution based on histograms. You construct the histograms using the `np.histogram` function and then you go through and normalize the PDF estimates found in the bins followed by creating the CDF by using the `cumsum` function. Now with these, you should be able to calculate the PDF, the CDF, the quantile function and virtually any other distribution function that you may need, even on new data.

So, how do you calculate quantities for new data? Let's break two functions down step by step: the PDF and the CDF/Quantile.

## Estimating PDFs

Recall, a histogram works by putting bins in intervals along the domain of your data. Then you calculate how many times you get values within those bins. The probability is the number of times you have data within that bin divided by the width of the bin. To make a density, you need to normalize the entire distribution so that it sums to 1. So in order to estimate the PDF for new data, you need to find where are the bins closest to your data. 


### Interpolation

One thing you can do is to interpolate. You find the support that is closest to your query points $X$ and then output the corresponding values for the estimated histogram values. Old school algebra, this looks like:

$$
\frac{x}{?} = \frac{\text{support}}{\text{hist pdf}}
$$

So in code, this translates to:

```python
x_pdf_est = np.interp(X_new, X_hist_bins, X_hist_pdf)
```

The numpy implementation has been shown to be quite fast. Sometimes I've used the scipy formula but apparently the numpy implementation [it's maginitudes faster](https://github.com/scipy/scipy/pull/4903#issuecomment-114259888) than the scipy implementation. However, I believe the scipy implementation has some extra goodies like extrapolation if you're outside the bounds of your distribution.

### Bisection Search

Alternatively, we could just do a bisection search. Now, it may not be as precise especially if we don't have enough bins, but it will be faster than interpolation. This works but 1) find the closest support values and then 2) use the corresponding values in the histogram PDF.

```python
# find the closest bins
X_bins_est = np.search_sorted(X_new, X_hist_bins)

# select the pdf of the bins
X_pdf_est = X_hist_pdf[X_bins_est]
```

## Estimating CDF / Quantiles

This is 

### CDFs


$$
\frac{X}{?} = \frac{Quantiles}{References}
$$

```python
X_uniform = np.interp(X, X_cdf, X_ref)
```

With the bisection search

```python
X_uniform = X_ref[np.search_sorted(X_cdf, X)]
```

### Quantiles

$$
\frac{X}{?} = \frac{References}{Quantiles}
$$

```python
X_approx = np.interp(X_uniform, X_references, X_quantiles)
```

With the bisection search

```python
X_approx = X_quantiles[np.search_sorted(X_references, X_uniform)]
```


## Application

In my application, I frequently work with normalizing flows, in particular Gaussianization. Essentially, you transform a univariate distribution to a Guassian distribution by computing the empirical CDF of the distribution followed by the Inverse of the Gaussian CDF. Now you can calculate the probability distribution using samples from the Gaussian distribution

## Other Implementations

### PyTorch

```python
def search_sorted(bin_locations, inputs, eps=1e-6):
    """
    Searches for which bin an input belongs to (in a way that is parallelizable and amenable to autodiff)
    """
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1
```

This function is a little difficult to understand. **Note**: this is what happens who you have no comments, no type hints and no documentation about the sizes or what the dimensions are...


So now, let's update the documentation so that it's clearer what's going on.

```python
def search_sorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Differentiable bisection search.
    
    Parameters
    ----------
    bin_locations : torch.Tensor, (n_samples, n_features)
    
    inputs : torch.Tensor, (n_samples, n_features)
    
    eps : float, default=1e-6
        regularization to be added
    
    Returns
    -------
    X_bins : torch.Tensor, (n_samples, n_features)
        corresponding bin locations of the inputs
    
    Example
    -------
    """

    return None
```


**Source**: [Pyro Library](http://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html) | [Neural Spline Flows]()