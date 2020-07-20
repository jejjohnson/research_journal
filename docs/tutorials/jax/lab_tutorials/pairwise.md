# Pairwise

$$
d(x,y) = \sum_{i=1}^N (x_i - y_i)^2
$$


---

#### Vector Representation

```python
def sqeuclidean_distances(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum( (x - y) ** 2)
```

---

#### Numpy Implementation

```python
d(x,y) = np.sqrt(np.dot(x, x) - 2.0 * np.dot(x, y) + np.dot(y, y))
```

* [Density Ratio Example](https://github.com/JohnYKiyo/density_ratio_estimation/blob/master/src/densityratio/densityratio.py#L187)
  > Pairwise euclidean distances with einsum, dot project and vmap.



---

#### Einsum

```python
XX = np.einsum("ik,ik->i", x, x)
YY = np.einsum("ik,ik->i", y, y)
XY = np.einsum("ik,jk->ij", x, y)

if not square:
    dists = np.sqrt(XX[:, np.newaxis] + YY[np.newaxis, :] - 2*XY)
else:
    dists = XX[:, np.newaxis] + YY[np.newaxis, :] - 2*XY
```

---

#### Dot Products

```python
XX = np.dot(x, x)
YY = np.dot(y, y)
XY = np.dot(x, y)

if not square:
    dists = np.sqrt(XX + YY - 2*XY)
else:
    dists = XX + YY - 2*XY
```

---

#### Pairwise Distances

```python
dists = jit(vmap(vmap(partial(dist, **arg), in_axes=(None, 0)), in_axes=(0, None)))
```


