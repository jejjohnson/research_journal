# RBF Parameters for HSIC Alignment Algorithm



## Theory

* What is Similarity?
* Why HSIC?
* The differences between HSIC
* The Problems with high-dimensional data


## Different Distributions




## Different Scorers


We are looking at different "HSIC scorers". They are:

---
**HSIC**

$$HSIC = \frac{1}{n(n-1)}\langle K_xH,K_yH \rangle_F$$

Notice: we have the centered kernels, $K_xH$ and no normalization.

---
**TKA** 

$$TKA = \frac{\langle K_x,K_y \rangle_F}{||K_x||_F||K_y||_F}$$

Notice: We have the uncentered kernels and a normalization factor.

---
**cTKA**

$$cTKA = \frac{\langle K_xH,K_yH \rangle_F}{||K_xH||_F||K_yH||_F}$$

Notice: We have the centered kernels and a normalization factor.



## Experiments


<center>

|                     | Options                      |
| ------------------- | ---------------------------- |
| Standardize         | Yes / No                     |
| Parameter Estimator | Mean, Median, Silverman, etc |
| Center Kernel       | Yes / No                     |
| Normalized Score    | Yes / No                     |

</center>