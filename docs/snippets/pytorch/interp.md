# Interpolating in PyTorch


Officially, there is not `interp` function in PyTorch. However, we do have the `searchsorted` function. This function performs a bisection


## Bisection Search



### Numpy Implementation

```python
np.searchsorted([1, 2, 3, 4, 5])
```

**Example**

```python
# interpolation method
new_y = np.interp(new_x, old_x, old_y)

# bisection method
new_y_ss = old_y[np.searchsorted(old_x, new_x, side='right')]
```


### PyTorch Implementation

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

**Source**: [Pyro Library](http://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html) | [Neural Spline Flows]()