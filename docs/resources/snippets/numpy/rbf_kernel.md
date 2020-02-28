# Optimized RBF kernel using `numexpr`

A fast implementation of the RBF Kernel using numexpr.

Using the fact that: ||x-y||<sup>2</sup> = ||x||<sup>2</sup> + ||y||<sup>2</sup> - 2 * x<sup>T</sup> * y

**Resources**
* Fast Implementation - [StackOverFlow](https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python)


```python
import numpy as np
import numexpr as ne
from sklearn.metrics import euclidean_distances
from sklearn.check

def rbf_kernel_ne(X, Y=None, length_scale=1.0, signal_variance=1.0):
    """This function calculates the RBF kernel. It has been optimized
    using some advice found online.
    
    Parameters
    ----------
    X : array, (n_samples x d_dimensions)
    
    Y : array, (n_samples x d_dimensions)
    
    length_scale : float, default: 1.0
    
    signal_variance : float, default: 1.0
    
    Returns
    -------
    K : array, (n_samples x d_dimensions)
    Resources
    ---------
    StackOverFlow: https://goo.gl/FXbgkj
    """
    X_norm = np.einsum('ij,ij->i', X, X)
    if Y is not None:
        Y_norm = np.einsum('ij,ij->i', Y, Y)
    else:
        Y = X
        Y_norm = X_norm

    K = ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
        'A': X_norm[:, None],
        'B': Y_norm[None, :],
        'C': np.dot(X, Y.T),
        'g': 1 / (2 * length_scale**2),
        'v': signal_variance
    })
    
    return K
    
def rbf_kernel(X, Y=None, signal_variance=1.0, length_scale=1.0):
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <rbf_kernel>`.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)

    K = euclidean_distances(X, Y, squared=True)
    K *= - 1 / (length_scale**2)
    np.exp(K, K)                # exponentiate K in-place
    K *= signal_variance        # multiply by signal_variance
    return K

```