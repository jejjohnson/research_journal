# RBIG 2.0 Ideas



* Neural Spline Flows - [github](https://github.com/bayesiains/nsf)
* Linear Rational Splines - [github](https://github.com/hmdolatabadi/LRS_NF)


---

$$\mathcal{X} \rightarrow \mathcal{Z}$$

**Option I**

* $F_\theta: (-\infty, \infty) \rightarrow [0,1]$ - CDF
* $\sigma^{-1}: [0,1] \rightarrow (-\infty, \infty)$ - Logit Function

**Option II**


* $F_\theta: (-\infty, \infty) \rightarrow [0,1]$
* $\text{Quantile}_{\mathcal{G}}^{-1}: [0,1] \rightarrow (-\infty, \infty)$ - Quantile Gaussian Function

---

### CDF Function

* Mixture of Gaussians
* Mixture of Logistics
* Linear Quadratic Splines
* Rational Quadratic Splines 
* NonLinear Squared Flows

### Logit Function

* Logit Function
* Gaussian Quantile Function

### Rotation

* HouseHolder Transforms
* SVD
* QR Decomposition
* LU Decomposition