# Rotation-Based Iterative Gaussianization (RBIG)

This is where I will outline some intricacies of the algorithms and how I will go about programming them.

--
## Motivation

The RBIG algorithm is a member of the density destructor family of methods. A density destructor is a generative model that seeks to transform your original data distribution, $\mathcal{X}$ to a base distribution, $\mathcal{Z}$ through an invertible tranformation $\mathcal{G}_\theta$, parameterized by $\theta$.

$$\begin{aligned}
x &\sim \mathcal P_x \sim \text{Data Distribution}\\
\hat z &= \mathcal{G}_\theta(x) \sim \text{Approximate Base Distribution}
\end{aligned}$$

Because we have invertible transforms, we can use the change of variables formula to get probability estimates of our original data space, $\mathcal{X}$ using our base distribution $\mathcal{Z}$. This is a well known formula written as:

$$p_x(x)=
p_{z}\left( \mathcal{G}_{\theta}(x) \right)
\left| \frac{\partial \mathcal{G}_{\theta}(x)}{\partial x} \right| =
 p_{z}\left( \mathcal{G}_{\theta}(x) \right) \left| \nabla_x \mathcal{G}(x) \right|
$$

If you are familiar with normalizing flows, you'll find some similarities between the formulations. Inherently, they are the same. However most (if not all) of the major methods of normalizing flows, they focus on the log-likelihood estimation of data $\mathcal{X}$. They seek to minimize this log-deteriminant of the Jacobian as a cost function. RBIG is different in this regard as it has a different objective. RBIG seeks to maximize the negentropy or minimize the total correlation.

Essentialy, RVIG is an algorithm that respects the name density destructor fundamental. We argue that by destroying the density, we maximize the entropy and destroy all redundancies within the marginals of the variables in question. From this formulation, this allows us to utilize RBIG to calculate many other IT measures which we highlight below.


$$\begin{aligned}
z &\sim \mathcal P_z \sim \text{Base Distribution}\\
\hat x &= \mathbf G_\phi(x) \sim \text{Approximate Data Distribution}
\end{aligned}$$



---
## Algorithm

> Gaussianization - Given a random variance $\mathbf x \in \mathbb R^d$, a Gaussianization transform is an invertible and differentiable transform $\mathcal \Psi(\mathbf)$ s.t. $\mathcal \Psi( \mathbf x) \sim \mathcal N(0, \mathbf I)$.

$$\mathcal G:\mathbf x^{(k+1)}=\mathbf R_{(k)}\cdot \mathbf \Psi_{(k)}\left( \mathbf x^{(k)} \right)$$

where:
* $\mathbf \Psi_{(k)}$ is the marginal Gaussianization of each dimension of $\mathbf x_{(k)}$ for the corresponding iteration.
* $\mathbf R_{(k)}$ is the rotation matrix for the marginally Gaussianized variable $\mathbf \Psi_{(k)}\left( \mathbf x_{(k)} \right)$



---
### Marginal (Univariate) Gaussianization

This transformation is the $\mathcal \Psi_\theta$ step for the RBIG algorithm.

In theory, to go from any distribution to a Gaussian distribution, we just need to

To go from $\mathcal P \rightarrow \mathcal G$

1. Convert the Data to a Normal distribution $\mathcal U$
2. Apply the CDF of the Gaussian distribution $\mathcal G$
3. Apply the inverse Gaussian CDF


So to break this down even further we need two key components

#### Marginal Uniformization

We have to estimate the PDF of the marginal distribution of $\mathbf x$. Then using the CDF of that estimated distribution, we can compute the uniform

$u=U_k(x^k)=\int_{-\infty}^{x^k}\mathcal p(x^k)dx^k$

This boils down to estimating the histogram of the data distribution in order to get some probability distribution. I can think of a few ways to do this but the simplest is using the histogram function. Then convert it to a [scipy stats rv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html) where we will have access to functions like pdf and cdf. One nice trick is to add something to make the transformation smooth to ensure all samples are within the boundaries.

* Example Implementation - [https://github.com/davidinouye/destructive-deep-learning/blob/master/ddl/univariate.py]()

From there, we just need the CDF of the univariate function $u(\mathbf x)$. We can just used the ppf function (the inverse CDF / erf) in scipy

#### Gaussianization of a Uniform Variable

Let's look at the first step: Marginal Uniformization. There are a number of ways that we can do this.

To go from $\mathcal G \rightarrow \mathcal U$

---
### Linear Transformation

This is the $\mathcal R_\theta$ step in the RBIG algorithm. We take some data $\mathbf x_i$ and apply some rotation to that data $\mathcal R_\theta (\mathbf x_i)$. This rotation is somewhat flexible provided that it follows a few criteria:

* Orthogonal
* Orthonormal
* Invertible

So a few options that have been implemented include:

* Independence Components Analysis (ICA)
* Principal Components Analysis (PCA)
* Random Rotations (random)

We would like to extend this framework to include more options, e.g. 

* Convolutions (conv)
* Orthogonally Initialized Components (dct)


The whole transformation process goes as follows:

$$\mathcal P \rightarrow \mathbf W \cdot \mathcal P \rightarrow U \rightarrow G$$

Where we have the following spaces:

* $\mathcal P$ - the data space for $\mathcal X$.
* $\mathbf W \cdot \mathcal P$ - The transformed space.
* $\mathcal U$ - The Uniform space.
* $\mathcal G$ - The Gaussian space.




  


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