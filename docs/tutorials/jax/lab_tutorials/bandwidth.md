# Bandwidth Approximation

## Median

```python
median = np.argsort(np.linalg.norm(np.abs(dists), ord=2, axis=-1))[int(dists.shape[0] / 2)]
bandwidth = factor * np.abs(dists)[median] ** 2 + 1e-5
```


