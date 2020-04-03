# RV Coefficient

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Lab: [Colab Notebook](https://colab.research.google.com/drive/19bJd_KNTSThZcxP1vnQOVjTLTOFLS9VG)

---


- [Notation](#notation)
- [Single Variables](#single-variables)
  - [Mean, Expectation](#mean-expectation)
    - [Empirical Estimate](#empirical-estimate)
  - [Variance](#variance)
    - [Empirical Estimate](#empirical-estimate-1)
  - [Covariance](#covariance)
    - [Empirical Estimate](#empirical-estimate-2)
  - [Correlation](#correlation)
    - [Empirical Estimate](#empirical-estimate-3)
  - [Root Mean Squared Error](#root-mean-squared-error)
- [Multi-Dimensional](#multi-dimensional)
  - [Variance](#variance-1)
    - [Self-Covariance](#self-covariance)
      - [Empirical Estimation](#empirical-estimation)
    - [Cross-Covariance](#cross-covariance)
      - [Empirical Estimation](#empirical-estimation-1)
    - [Root Mean Squared Vector Difference](#root-mean-squared-vector-difference)
  - [Summarizing Multi-Dimensional Information](#summarizing-multi-dimensional-information)
    - [Sample Space](#sample-space)
    - [Equivalence](#equivalence)
- [Supplementary](#supplementary)

---

## Notation

* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{x}}$ are samples from a multidimentionsal r.v. $\mathcal{X}$
* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{y}}$ are samples from a multidimensional r.v. $\mathcal{Y}$
* $\Sigma \in \mathbb{R}^{N \times N}$ is a covariance matrix.
  * $\Sigma_\mathbf{x}$ is a kernel matrix for the r.v. $\mathcal{X}$
  * $\Sigma_\mathbf{y}$ is a kernel matrix for the r.v. $\mathcal{Y}$
  * $\Sigma_\mathbf{xy}$ is the population covariance matrix between $\mathcal{X,Y}$
* $tr(\cdot)$ - the trace operator
* $||\cdot||_\mathcal{F}$ - Frobenius Norm
  * $||\cdot||_\mathcal{HS}$ - Hilbert-Schmidt Norm 
* $\tilde{K} \in \mathbb{R}^{N \times N}$ is the centered kernel matrix.

---

## Single Variables

Let's consider a single variable $X \in \mathbb{R}^{N \times 1}$ which represents a set of samples of a single feature. 


### Mean, Expectation

The first order measurement is the mean. This is the expected/average value that we would expect from a r.v.. This results in a scalar value


#### Empirical Estimate

$$\mu(x)=\frac{1}{N}\sum_{i=1}x_i$$

### Variance

The first measure we need to consider is the variance. This is a measure of spread.


#### Empirical Estimate


$$
\begin{aligned}
\sigma_x^2 
&= \frac{1}{n-1} \sum_{i=1}^N(x_i-x_\mu)^2
\end{aligned}
$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

```python
# remove mean from data
X_mu = X.mean(axis=0)

# ensure it is 1D
var = (X - X_mu[:, None]).T @ (X - X_mu[:, None])
```

</details>

---

### Covariance


The first measure we need to consider is the covariance. This can be used for a single variable $X \in \mathbb{R}^{N \times 1}$ which represents a set of samples of a single feature. We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. This results in a scalar value , $\mathbb{R}$. We can write this as:

$$
\begin{aligned}
\text{cov}(\mathbf{x,y}) 
&= \mathbb{E}\left[(\mathbf{x}-\mu_\mathbf{x})(\mathbf{y}-\mu_\mathbf{y}) \right] \\
&= \mathbb{E}[\mathbf{xy}] - \mu_\mathbf{x}\mu_\mathbf{y}
\end{aligned}
$$

<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

$$
\begin{aligned}
\text{cov}(\mathbf{x,y})  &= \mathbb{E}\left((\mathbf{x}-\mu_\mathbf{x})(\mathbf{y}-\mu_\mathbf{y}) \right) \\
&= \mathbb{E}\left[\mathbf{xy} - \mu_\mathbf{x} Y - \mathbf{x}\mu_\mathbf{y} + \mu_\mathbf{x}\mu_y \right] \\
&=  \mathbb{E}[\mathbf{xy}] - \mu_\mathbf{x}  \mathbb{E}[\mathbf{x}] -  \mu_y\mathbb{E}[\mathbf{y}] + \mu_\mathbf{x}\mu_y \\
&=  \mathbb{E}[\mathbf{xy}] - \mu_\mathbf{x}\mu_y
\end{aligned}
$$
</details>

This will result in a scalar value $\mathbb{R}^+$ that ranges from $(-\infty, \infty)$. This number is affected by scale so we can different values depending upon the scale of our data, i.e. $\text{cov}(\mathbf{x,y}) \neq \text{cov}(\alpha \mathbf{x}, \beta \mathbf{x})$ where $\alpha, \beta \in \mathbb{R}^{+}$


#### Empirical Estimate

We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. We can write this as:

$$\text{cov}(\mathbf{x,y})  = \frac{1}{n-1} \sum_{i=1}^N (x_i - x_\mu)(y_i - y_\mu)$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ Y
```
</details>

---

### Correlation


This is the normalized version of the covariance measured mentioned above. This is done by dividing the covariance by the product of the standard deviation of the two samples X and Y.  

$$\rho(\mathbf{x,y})=\frac{\text{cov}(\mathbf{x,y}) }{\sigma_x \sigma_y}$$

This results in a scalar value $\mathbb{R}$ that lies in between $[-1, 1]$. When  $\rho=-1$, there is a negative correlation and when $\rho=1$, there is a positive correlation. When $\rho=0$ there is no correlation.

#### Empirical Estimate

So the formulation is:

$$\rho(\mathbf{x,y}) = \frac{\text{cov}(\mathbf{x,y}) }{\sigma_x \sigma_y}$$

With this normalization, we now have a measure that is bounded between -1 and 1. This makes it much more interpretable and also invariant to isotropic scaling, $\rho(X,Y)=\rho(\alpha X, \beta Y)$ where $\alpha, \beta \in \mathbb{R}^{+}$

---

### Root Mean Squared Error

This is a popular measure for measuring the errors between two datasets. More or less, it is a covariance measure that penalizes higher deviations between the datasets.

$$RMSE(X,Y)=\sqrt{\frac{1}{N}\sum_{i=1}^N \left((x_i - \mu_x)-(y_i - \mu_i)\right)^2}$$


---

## Multi-Dimensional

For all of these measures, we have been under the assumption that $\mathbf{x,y} \in \mathbb{R}^{N \times 1}$. However, we may have the case where we have multivariate datasets in $\mathbb{R}^{N \times D}$. In this case, we need methods that can handle multivariate inputs. 

### Variance

#### Self-Covariance

So now we are considering the case when we have multidimensional vectors. If we think of a variable $X \in \mathbb{R}^{N \times D}$ which represents a set of samples with multiple features. First let's consider the variance for a multidimensional variable. This is also known as the covariance because we are actually finding the cross-covariance between itself.

$$
\begin{aligned}
\text{Var}(X) 
&= \mathbb{E}\left[(X-\mu_x)^2 \right] \\
\end{aligned}
$$


<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

$$
\begin{aligned}
\text{Var}(X) &= \mathbb{E}\left((X-\mu_x)(X-\mu_y) \right) \\
&= \mathbb{E}\left(XX - \mu_XX - X\mu_X + \mu_X\mu_X \right) \\
&=  \mathbb{E}(XX) - \mu_x  \mathbb{E}(X) -  \mathbb{E}(X)\mu_X + \mu_x\mu_X \\
&=  \mathbb{E}(X^2) - \mu_X^2
\end{aligned}
$$
</details>

To simplify the notation, we can write this as:

$$\Sigma_\mathbf{x} = \text{cov}(\mathbf{x,x})$$

* A completely diagonal linear kernel (Gram) matrix means that all examples are uncorrelated (orthogonal to each other).
* Diagonal kernels are useless for learning: no structure found in the data.


##### Empirical Estimation

This shows the joint variation of all pairs of random variables.

$$\Sigma_\mathbf{x} = \mathbf{x}^\top \mathbf{x}$$


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ X
```
</details>

---

#### Cross-Covariance

 We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. We can write this as:

$$
\begin{aligned}
\text{cov}(\mathbf{x,y}) &= \mathbb{E}\left[(\mathbf{x}-\mu_\mathbf{x})(\mathbf{y}-\mu_\mathbf{y}) \right] \\
&= \mathbb{E}[\mathbf{xy}] - \mu_\mathbf{x}\mu_\mathbf{y}
\end{aligned}
$$



<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

$$
\begin{aligned}
C(X,Y) &= \mathbb{E}\left((X-\mu_x)(Y-\mu_y) \right) \\
&= \mathbb{E}\left(XY - \mu_xY - X\mu_y + \mu_x\mu_y \right) \\
&=  \mathbb{E}(XY) - \mu_x  \mathbb{E}(X) -  \mathbb{E}(X)\mu_y + \mu_x\mu_y \\
&=  \mathbb{E}(XY) - \mu_x\mu_y
\end{aligned}
$$
</details>

This results in a scalar value which represents the similarity between the samples. There are some key observations of this measure.


##### Empirical Estimation

This shows the joint variation of all pairs of random variables.

$$\Sigma_\mathbf{xy} = \mathbf{x}^\top \mathbf{y}$$


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ X
```
</details>

**Observations**
* A completely diagonal covariance matrix means that all features are uncorrelated (orthogonal to each other).
* Diagonal covariances are useful for learning, they mean non-redundant features!

---

#### Root Mean Squared Vector Difference

* [A diagram for evaluating multiple aspects of model performance insimulating vector fields](https://www.geosci-model-dev.net/9/4365/2016/gmd-9-4365-2016.pdf) - Xu et. al. (2016)



---
### Summarizing Multi-Dimensional Information

Recall that we now have self-covariance matrices $\Sigma_\mathbf{x}$ and cross-covariance matrices $\Sigma_\mathbf{xy}$ which are $\mathbb{R}^{D \times D}$. This is very useful as it captures the structure of the overall data. However, if we want to summarize the statistics, then we need some methods to do so. The matrix norm, in particular the Frobenius Norm (aka the Hilbert-Schmidt Norm) to effectively summarize content within this covariance matrix. It's defined as:

$$||\Sigma_\mathbf{xy}||_{\mathcal{F}}^2 = \sum_i \lambda_i^2 = \text{tr}\left( \Sigma_\mathbf{xy}^\top \Sigma_\mathbf{xy} \right)$$
 
 Essentially this is a measure of the covariance matrix power or "essence" through its eigenvalue decomposition. Note that this term is zero iff $\mathbf{x,y}$ are independent and greater than zero otherwise. Since the covariance matrix is a second-order measure of the relations, we can only summarize the the second order relation information. But at the very least, we now have a scalar value in $\mathbb{R}$ that summarizes the structure of our data.

In the context of matrices, we can normalize this value by the norm of the self-covariance matrices like so:

$$\rho V (\mathbf{x,y}) = \frac{\langle \Sigma_\mathbf{xy}, \Sigma_\mathbf{xy} \rangle_F}{||\Sigma_\mathbf{xx}||_F \; || \Sigma_\mathbf{yy}||_F}$$

This results in the $\rho$V-Coefficient which is analogous to the Pearson correlation coefficient $\rho$.

!> **Note** We assume that the data is column centered (aka we have removed the mean from the features).



**Observations**
* HSIC norm of the covariance only detects second order relationships. More complex (higher-order, nonlinear) relations cannot be captured


Let's add $N$ independent realizations to the samples. This gives us a vector for each of the observations. So, let $\mathbf{X} \in \mathbb{R}^{N \times D_x}$ and $\mathbf{Y} \in \mathbb{R}^{N \times D_y}$. We assume that they are column-centered (aka remove the mean from the features). So, we can write the $S_{\mathbf{XY}}= \frac{1}{n-1}\mathbf{X^\top Y}$

$$
\begin{aligned}
\text{RV}(\mathbf{X,Y})
&= 
\frac{tr\left( S_{\mathbf{XY}}S_{\mathbf{XY}} \right)}{\sqrt{tr\left( S_{\mathbf{XX}}^2 \right) tr\left( S_{\mathbf{YY}}^2 \right)}}
\end{aligned}
$$


#### Sample Space


We can also consider the case where the correlations can be measured between samples and not between features. So we can create cross product matrices: $\mathbf{W}_\mathbf{X}=\mathbf{XX}^\top \in \mathbb{R}^{N \times N}$ and $\mathbf{W}_\mathbf{Y}=\mathbf{YY}^\top \in \mathbb{R}^{N \times N}$. Just like the feature space, we can use the Hilbert-Schmidt (HS) norm, $||\cdot||_{F}$ to measure proximity. 

$$
\begin{aligned}
\langle {W}_\mathbf{x}, {W}_\mathbf{y} \rangle 
&= 
tr \left( \mathbf{xx}^\top \mathbf{yy}^\top \right) \\
&= 
\sum_{i=1}^{D_x} \sum_{j=1}^{D_y} cov^2(\mathbf{x}_{d_i}, \mathbf{y}_{d_j})
\end{aligned}
$$

And like the above mentioned $\rho V$, we can also calculate a correlation measure using the sample space.

$$
\begin{aligned}
\rho V(\mathbf{x,y}) 
&=
\frac{\langle \mathbf{W_x, W_y}\rangle_F}{||\mathbf{W_x}||_F \; ||\mathbf{W_y}||_F} \\
\end{aligned}
$$
 


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>


**Sample Space**

Recall that the $\rho V$ can be written in terms of trace operations.

$$\rho V(\mathbf{x,y})  = \frac{tr\left( \mathbf{xx}^\top \mathbf{yy}^\top \right)}{\sqrt{\text{tr}\left( \mathbf{xx}^\top \right)^2 \text{tr}\left( \mathbf{yy}^\top \right)^2}}$$

This is very easy to compute in practice. One just needs to calculate the Frobenius Norm (Hilbert-Schmidt Norm) of a covariance matrix This boils down to computing the trace of the matrix multiplication of two matrices: $tr(C_{xy}^\top C_{xy})$. So in algorithmically that is:

```python
hsic_score = np.sqrt(np.trace(C_xy.T * C_xy))
```
We can make this faster by using the `sum` operation

```python
# Numpy
hsic_score = np.sqrt(np.sum(C_xy * C_xy))
# PyTorch
hsic_score = (C_xy * C_xy).sum().sum()
```

**Refactor**

There is a built-in function to be able to to speed up this calculation by a magnitude.

```python
hs_score = np.linalg.norm(C_xy, ord='fro')
```

and in PyTorch

```python
hs_score = torch.norm(C_xy, p='fro)
```
</details>


#### Equivalence

It turns out, for the linear case, when using the Frobenius norm to summarize the pairwise comparisons, comparing features is the same as comparing samples. For example, the norm of the covariance operator for the features and samples are equivalent:

$$||\Sigma_{\mathbf{xy}}||_F^2 = 
\langle \mathbf{W_x,W_y} \rangle_F
$$

We get the same for the $\rho V$ case.

$$\frac{ ||\Sigma_{\mathbf{xy}}||_F^2}{||\Sigma_\mathbf{x}||_F ||\Sigma_\mathbf{y}||_F} = 
\frac{ \langle \mathbf{W_x,W_y} \rangle}{||\mathbf{W_x}||_F ||\mathbf{W_y}||_F}
$$

So what does this mean? Well, either method is fine. But you should probably choose one depending upon the computational resources available. For example, if you have more samples than features, then choose the feature space representation. On the other hand, if you have more features than samples, then choose the sample space representation.

!> **Linear Only** This method only works for the linear case. There are some nonlinear transformations (called kernels) that one can use, but those will yield different values between feature space and sample space.





---

## Supplementary

* Common Statistical Tests are Linear Models (or: How to Teach Stats) - Jonas Kristoffer Lindelov - [notebook](https://eigenfoo.xyz/tests-as-linear/) | [rmarkdown](https://lindeloev.github.io/tests-as-linear/)
* Correlation vs Regression - Asim Jana - [blog](https://www.datasciencecentral.com/profiles/blogs/difference-between-correlation-and-regression-in-statistics)
* RealPython
  * Numpy, SciPy and Pandas: Correlation with Python - [blog](https://realpython.com/numpy-scipy-pandas-correlation-python/)
* Correlation and Lag for Signals - [notebook](https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html)
* [Understanding the Covariance Matrix](https://datascienceplus.com/understanding-the-covariance-matrix/)
* [Numpy Vectorized method for computing covariance with population means](https://stackoverflow.com/questions/34235452/numpy-vectorised-method-for-computing-covariance-with-population-means-for-surv)
* Eric Marsden
  * [Modeling Correlations in Python](https://www.slideshare.net/EricMarsden1/modelling-correlations-using-python)
  * [Regression Analysis in Python](https://www.slideshare.net/EricMarsden1/regression-analysis-using-python)