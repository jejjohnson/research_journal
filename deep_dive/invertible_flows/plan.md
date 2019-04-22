# Plan for Notebook Series


---
### Equation Breakdown

**Perspective Routes**

Destructive Transformation:

$$x \overset{\bf f}{\rightarrow} z$$

Constructive/Generative Transformation:
$$z \overset{\bf g}{\rightarrow} z$$

#### Concrete Equations

$$\begin{aligned}
x &\sim \mathcal P_x \sim \text{Data Distribution}\\
\hat z &= \mathbf D_\theta(x) \sim \text{Approximate Base Distribution}
\end{aligned}$$

$$\begin{aligned}
z &\sim \mathcal P_z \sim \text{Base Distribution}\\
\hat x &= \mathbf G_\phi(x) \sim \text{Approximate Data Distribution}
\end{aligned}$$

#### GANs vs VAEs vs IFs

[blog](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* [VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)


#### Base Distributions


#### Functions

* Non-Invertible (VAEs)
* Invertible:
  * Inverse Sampling Theorem (Uniform and PDFs)
  * Change of Variables

---
**Core Topics**

* Parametric Gaussianization
* Deep Density Destructors
* Information Theory Measures
* Generalized Divisive Normalization


**Supplementary**

* Inverse Sampling Theorem
* Change of Variables Formula
* Entropy
* NegEntropy

**Formulations**

* Deep Density Destructor
* Normalizing Flows
* Parametric Gaussianization
* Rotation-Based Iterative Gaussianization (RBIG)
* Generalized Divisive Normalization (GDN)