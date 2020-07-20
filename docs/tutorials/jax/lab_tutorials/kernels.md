# Kernel Matrices


---

### Linear Kernel

```python
def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    return np.inner(x, y)
```

---

### Polynomial Kernel

```python
def polynomial_kernel(params, x: np.ndarray, y: np.ndarray) -> float:
    return np.power(params['alpha'] * np.inner(x, y)), params['degree']
```

---

### Sigmoid Kernel

```python
def sigmoid_kernel(params, x: np.ndarray, y: np.ndarray) -> float:
    return np.tanh(params['alpha'], np.inner(x, y) + c)
```

---

### RBF Kernel

$$
k(x,y) = \exp(-\gamma ||x-y||_2^2)
$$

```python
def rbf_kernel(params, x: np.ndarray, y: np.ndarray) -> float:
    """The RBF Kernel"""
    # calculate the kernel
    return np.exp(- params['gamma'] * (sqeuclidean_distances(x, y)) )
```

---

### ARD Kernel


```python
def ard_kernel(params, x, y):
    """The RBF Kernel"""
    
    # scale the data
    x = np.divide(x, params['length_scale'])
    y = np.divide(y, params['length_scale'])

    # calculate the kernel
    return np.exp(- (sqeuclidean_distances(x, y)) )
```

---

### Gram Matrix


**Method I** - single call

```python
def gram(kernel_func, params, X, Y=None):
    if Y is None:
        return vmap(lambda x: vmap(lambda y: kernel_func(params, x, y))(X))(X)
    else:
        return vmap(lambda x: vmap(lambda y: kernel_func(params, x, y))(Y))(X)

```

**Method II** - multiple calls

```python
# Covariance Matrix
def covariance_matrix(kernel_func, x, y):
    mapx1 = jax.vmap(lambda x, y: kernel_func(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)
```