# Add every n values in array

This is the case where you want to add every 3.

So for example:

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

We want the following if we add every 3 values:

```python
b = [1+2+3, 4+5+6, 7+8+9]
b = [6, 14, 24]
```

---

We can do this by reshaping the array


1. Make sure it's at minimum, a 2D array.

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
print(a.shape)
```

```
(9,1)
```

2. Reshape via a chunksize

```python
chunk_size = 3
a = a.reshape(-1, chunk_size, a.shape[1])
print(a.shape)
```

```
(3, 3, 1)
```

3. Sum the middle column.

```python
a = a.sum(1)
print(a.shape)
print(a.squeeze())
```

```
(3, 1)
[ 6 15 24]
```

**Note**: the size of a has to be divisible by the chunk-size. So in our case 9/3=3 so we're good. But this wouldn't work for 10/3 because we have some remainder.

**Source**: [StackOverFlow](https://stackoverflow.com/questions/52240535/numpy-undestand-how-to-sum-every-n-values-in-an-array-matrix)

---

#### In One Shot


We can do this in one shot using the numpy built-in function:

```python
import numpy as np

chunk_size = 3

# 1d arrays
a = np.ones((9))
a = np.add.reduceat(a, np.arange(0, len(a), chunk_size))

# n-d arrays
a = np.ones((9,9))
a = np.add.reduceat(a, np.arange(0, len(a), chunk_size), axis=0)
```