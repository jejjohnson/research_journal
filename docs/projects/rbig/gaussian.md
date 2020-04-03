# Gaussian Distribution



### **PDF**

$$f(X)=
\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}
\text{exp}\left( -\frac{1}{2} (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

### **Likelihood**

$$- \ln L = \frac{1}{2}\ln|\Sigma| + \frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x - \mu) + \frac{D}{2}\ln 2\pi $$

### Alternative Representation

$$X \sim \mathcal{N}(\mu, \Sigma)$$

where $\mu$ is the mean function and $\Sigma$ is the covariance. Let's decompose $\Sigma$ as with an eigendecomposition like so

$$\Sigma = U\Lambda U^\top = U \Lambda^{1/2}(U\Lambda^{-1/2})^\top$$

Now we can represent our Normal distribution as:

$$X \sim \mu + U\Lambda^{1/2}Z$$



where:

* $U$ is a rotation matrix
* $\Lambda^{-1/2}$ is a scale matrix
* $\mu$ is a translation matrix
* $Z \sim \mathcal{N}(0,I)$

or also

$$X \sim \mu + UZ$$

where:

* $U$ is a rotation matrix
* $\Lambda$ is a scale matrix
* $\mu$ is a translation matrix
* $Z_n \sim \mathcal{N}(0,\Lambda)$


#### Reparameterization

So often in deep learning we will learn this distribution by a reparameterization like so:

$$X = \mu + AZ $$

where:

* $\mu \in \mathbb{R}^{d}$
* $A \in \mathbb{R}^{d\times l}$
* $Z_n \sim \mathcal{N}(0, I)$
* $\Sigma=AA^\top$ - the cholesky decomposition



---
### **Entropy**

**1 dimensional**

$$H(X) = \frac{1}{2} \log(2\pi e \sigma^2)$$

**D dimensional**
$$H(X) = \frac{D}{2} + \frac{D}{2} \ln(2\pi) + \frac{1}{2}\ln|\Sigma|$$


### **KL-Divergence (Relative Entropy)**

$$
KLD(\mathcal{N}_0||\mathcal{N}_1) = \frac{1}{2}
 \left[ 
 \text{tr}(\Sigma_1^{-1}\Sigma_0) + 
 (\mu_1 - \mu_0)^\top \Sigma_1^{-1} (\mu_1 - \mu_0) -
D + \ln \frac{|\Sigma_1|}{\Sigma_0|}
\right]
$$

if $\mu_1=\mu_0$ then:

$$
KLD(\Sigma_0||\Sigma_1) = \frac{1}{2} \left[ 
\text{tr}(\Sigma_1^{-1} \Sigma_0)  - D  + \ln \frac{|\Sigma_1|}{|\Sigma_0|} \right]
$$

**Mutual Information**

$$I(X)= - \frac{1}{2} \ln | \rho_0 |$$

where $\rho_0$ is the correlation matrix from $\Sigma_0$.

$$I(X)$$
