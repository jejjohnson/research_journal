

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

#### Testing

Here we have a few tests that we can do:

* DCT components are orthogonal
* PCA works
  * Eigenvalues are descending
* PCA + Whitening Works
  * Eigenvalues are descending
  * Data is white [all(abs(np.cov(pca.transform(data))< 1e-6>)]
* Test the non-symmetric version (?)

```python

def test_dct(self):

    # Initialize Linear Transformation Class and DCT components
    dct = LinearTransformation(basis='DCT', conv=16)

    # Make sure the DCT basis is orthogonal
    self.assertTrue(all(abs(dct.W, dct.W.T) - np.eye(256)) < 1e-10)

    # The Jacobian should be zero
    X_rand = np.random.randn(16, 10)
    self.assertTrue(all(abs(dct.logjacobian(X_rand), dct.W.T) - np.eye(256) < 1e-10))

def test_pca(self):

    # Get Test Data
    X_rand = np.random.randn(16, 256)

    covr = np.cov(X_rand)
    
    data = np.linalg.cholesky(covr) @ np.random.randn(16, 10000)

    # Initialize Linear Transformation with PCA
    pca = LinearTransformation(basis='PCA')

    # Make sure eigenvalues descend

    #


    # Make sure data is white

def test_pca_whitening(self):

```

---
## Supplementary

---
### Boundary Issues



#### PDF Estimation under arbitrary transformation

Let $\mathbf x \in \mathbb R^d$ be a r.v. with a PDF, $\mathcal P_x (\mathbf x)$. Given some bijective, differentiable transform $\mathbf x$ and $\mathbf y$ using $\mathcal G:\mathbb R^d \rightarrow \mathbb R^d$, $\mathbf y = \mathcal G(\mathbf x)$, we can use the change of variables formula to calculate the determinant:

$$\mathcal{P}_x( \mathbf x)=
\mathcal{P}_{y}\left( \mathcal{G}_{\theta}( \mathbf x) \right)
\left| \frac{\partial \mathcal{G}_{\theta}(\mathbf x)}{\partial \mathbf x} \right|$$

$$\mathcal{P}_x( \mathbf x)=
\mathcal{P}_{y}\left( \mathcal{G}_{\theta}( \mathbf x) \right) \cdot
\left| \nabla_{\mathbf x} \cdot \mathcal{G}_{\theta}(\mathbf x) \right|$$

In the case of Gaussianization, we can calculate $\mathcal P (\mathbf x)$ if the Jacobian is known since

#### Iterative Gaussianization Transform is Invertible

Given a Gaussianization transform:

$$\mathcal G:\mathbf x^{(k+1)}=\mathbf R_{(k)}\cdot\mathbf \Psi_{(k)}\left( \mathbf x^{(k)} \right)$$

by simple manipulation, the inversion transform is:

$$\mathcal G^{-1}:\mathbf x^{(k)}=\mathbf \Psi_{(k)}^{-1}\left( \mathbf R_{(k)}^{-1} \cdot \mathbf x^{(k)} \right)$$

**Note**: If $\mathbf R_{(k)}^{-1}$ is orthogonal, then $\mathbf R_{(k)}^{-1} = \mathbf R_{(k)}^{\top}$. So we can simplify our transformation like so:

$$\mathcal G^{-1}:\mathbf x^{(k)}=\mathbf \Psi_{(k)}^{-1}\left( \mathbf R_{(k)}^{\top} \cdot \mathbf x^{(k)} \right)$$

iff $\mathbf R_{(k)}^{-1}$ is orthogonal or (orthonormal vectors).

**Note 2**: to ensure that $\mathbf \Psi_{(k)}$ is invertible, we need to be sure that the PDF support is connected. So the domain is continuous and there are no disjoint spaces (**???**).

---
## References

**Algorithm**

* Multivariate Gaussianization for Data Proceessing - [Prezi](https://www-n.oca.eu/aferrari/MAHI/GCampsMAHI12.pdf)
* Nonlineear Extraction of 'IC' of elliptically symmetric densities using radial Gaussianization - Lyu et. al. (2008) - 

**Code**

* Real-NVP Implementation - [PyTorch](https://github.com/chrischute/real-nvp/blob/master/models/real_nvp/real_nvp.py)
* Normalizing Flows with [PyTorch](https://github.com/acids-ircam/pytorch_flows)
* Radial Gaussianization - [Python](https://github.com/spencerkent/rg-toolbox)
* [PyTorch GDN](https://github.com/jorge-pessoa/pytorch-gdn)
* Good RBIG Implementations - [Transforms](https://github.com/lucastheis/isa/tree/master/code/transforms) | [Models](https://github.com/lucastheis/isa/tree/master/code/models)
* Histogram Estimation
  * [Scipy Use](https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/univariate.py#L46)
  * [Scipy Function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html)
  * [HistogramUniverateDensity](https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/univariate.py#L365)