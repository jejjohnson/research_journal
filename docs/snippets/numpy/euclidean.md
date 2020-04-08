# Efficient Euclidean Distance Calculation - Numpy Einsum 

This script uses numpy's einsum function to calculate the euclidean distance.
Resources:

```python
import numpy as np

def euclidean_distance_einsum(X, Y):
    """Efficiently calculates the euclidean distance
    between two vectors using Numpys einsum function.
    
    Parameters
    ----------
    X : array, (n_samples x d_dimensions)
    Y : array, (n_samples x d_dimensions)
    
    Returns
    -------
    D : array, (n_samples, n_samples)
    """
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)
#    XY = 2 * np.einsum('ij,kj->ik', X, Y)
    XY = 2 * np.dot(X, Y.T)
    return  XX + YY - XY
```

An alternative way per [stackoverflow](https://stackoverflow.com/questions/32154475/einsum-and-distance-calculations) would be to do it in one shot.

**Sources**

* [How to calculate euclidean distance between pair of rows of a numpy array](https://stackoverflow.com/questions/43367001/how-to-calculate-euclidean-distance-between-pair-of-rows-of-a-numpy-array)
* [Calculate Distance between numpy arrays](https://stackoverflow.com/questions/40996957/calculate-distance-between-numpy-arrays)
* [einsum and distance calculations](https://stackoverflow.com/questions/32154475/einsum-and-distance-calculations)
* [How can the Euclidean distance be calculated with NumPy?](https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy)
* [Using Python numpy einsum to obtain dot product between 2 Matrices](https://stackoverflow.com/questions/45896939/using-python-numpy-einsum-to-obtain-dot-product-between-2-matrices)
* [High-Performance computation in Python | NumPy](https://semantive.com/blog/high-performance-computation-in-python-numpy-2/)