# Rotation

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)
* Colab Notebook: [Notebook](https://colab.research.google.com/drive/1j5GSPGHhpvKV2MiJD5Sjmlb5J3Re1SMY)

## Main Idea

### Rotation Matrix


## Foward Transformation

## Reverse Transformation

## Jacobian

The deteriminant of an orthogonal matrix is 1. 

**Proof**:

There are a series of transformations that can be used to prove this:

$$
\begin{aligned}
1 &= \det(\mathbf{I}) \\
&= \det (\mathbf{R^\top R}) \\
&= \det (\mathbf{R^\top}) \det(\mathbf{R}) \\
&= \det(\mathbf{R})^2 \\
\end{aligned}
$$

Therefore, we can conclude that the $\det(\mathbf{R})=1$.

### Log Jacobian


As shown above, the log determinant jacobian of an orthogonal matrix is 1. So taking the log of this is simply zero.

$$
\log(\det(\mathbf{R})) = \log(1) = 0
$$

### Decompositions


#### QR Decomposition

$$A=QR$$

where 
* $A \in \mathbb{R}^{N \times M}$
* $Q \in \mathbb{R}^{N \times N}$ is orthogonal
* $R \in \mathbb{R}^{N \times M}$ is upper triangular

#### Singular Value Decomposition

Finds the singular values of the matrix.

$$A=U\Sigma V^\top$$

where:
* $A \in \mathbb{R}^{N \times M}$
* $U \in \mathbb{R}^{N \times K}$ is unitary
* $\Sigma \in \mathbb{R}^{K \times K}$ are the singular values
* $V^\top \in \mathbb{R}^{K \times M}$ is unitary

#### Eigendecomposition

Finds the singular values of a symmetric matrix

$$A_S=Q\Lambda Q^\top$$

where:
* $A_S \in \mathbb{R}^{N \times N}$
* $Q \in \mathbb{R}^{N \times K}$ is unitary
* $\Lambda \in \mathbb{R}^{K \times K}$ are the singular values
* $Q^\top \in \mathbb{R}^{K \times N}$ is unitary

#### Polar Decomposition

$$A_S=QS$$

where:
* $A_S \in \mathbb{R}^{N \times N}$
* $Q \in \mathbb{R}^{N \times K}$ is unitary
* $\Lambda \in \mathbb{R}^{K \times K}$ are the singular values
* $Q^\top \in \mathbb{R}^{K \times N}$ is unitary


---
#### Initializing

We have a intialization step where we compute $\mathbf W$ (the transformation matrix). This will be in the fit method. We will need the data because some transformations depend on $\mathbf x$ like the PCA and the ICA.

```python
class LinearTransform(BaseEstimator, TransformMixing):
    """
    Parameters
    ----------

    basis : 
    """
    def __init__(self, basis='PCA', conv=16):
        self.basis = basis
        self.conv = conv

    def fit(self, data):
        """
        Computes the inverse transformation of 
                z = W x

        Parameters
        ----------
        data : array, (n_samples x Dimensions)
        """

        # Check the data

        # Implement the transformation
        if basis.upper() == 'PCA':
            ...
        elif basis.upper() == 'ICA':
            ...
        elif basis.lower() == 'random':
            ...
        elif basis.lower() == 'conv':
            ...
        elif basis.upper() == 'dct':
            ...
        else:
            Raise ValueError('...')
        
        # Save the transformation matrix
        self.W = ...

        return self
```

#### Transformation

We have a transformation step:

$$\mathbf{z=W\cdot x}$$

where:
* $\mathbf W$ is the transformation
* $\mathbf x$ is the input data
* $\mathbf y$ is the final transformation

```python
def transform(self, data):
    """
    Computes the inverse transformation of 
            z = W x

    Parameters
    ----------
    data : array, (n_samples x Dimensions)
    """
    return data @ self.W
```

#### Inverse Transformation

We also can apply an inverse transform.

```python
def inverse(self, data):
    """
    Computes the inverse transformation of 
    z = W^-1 x

    Parameters
    ----------
    data : array, (n_samples x Dimensions)

    Returns
    -------
    
    """
    return data @ np.linalg.inv(self.W)
```

#### Jacobian

Lastly, we can calculate the Jacobian of that function. The Jacobian of a linear transformation is
just 

````python
def logjacobian(self, data=None):
    """

    """
    if data is None:
        return np.linalg.slogdet(self.W)[1]
    
    return np.linalg.slogdet(self.W)[1] + np.zeros([1, data.shape[1]])
````

#### Log Likelihood (?)