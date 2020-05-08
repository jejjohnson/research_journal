# Normalizing Flows Literature



---

## Training

* Can use Reverse-KL
* Can use Forward-KL (aka Maximum Likelihood Estimation)
* Generally possible to sample from the model

**Maximum Likelihood Training**

$$
\log p_\theta(\mathbf{x}) = \log p_\mathbf{z}(f(\mathbf{x})) + \log \left| \det \nabla_\mathbf{x} f_\theta(\mathbf{x}) \right|
$$

**Stochastic Gradients**

$$
\nabla_\theta \mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \log p_\theta(\mathbf{x}) \right] =
\mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \nabla_\theta \log p_\theta(\mathbf{x})  \right]
$$

* Stochastic
* Scales to Large Datasets
* Converges to True minimum
* Large body of supportive software

---

## Summarizing

> Almost all papers are trying to do some form of creating a clever jacobian so that it is relatively cheap to calculate and work with.

I like the slides in [this](https://www.cs.toronto.edu/~rtqichen/pdfs/residual_flows_slides.pdf) presentation which attempts to summarize the different methods and how they are related.

| Jacobian Type          | Methods                          |
| ---------------------- | -------------------------------- |
| Determinant Identities | Planar NF, Sylvester NF          |
| Coupling Blocks        | NICE, Real NVP, GLOW             |
| AutoRegressive         | Inverse AF, Neural AF, Masked AF |
| Unbiased Estimation    | FFJORD, Residual Flows           |
| Diagonal               | Gaussianization Flows, GDN       |


---

## Automatic Differentiation

According to a talk by Ricky Chen:

> For a full Jacobian, need $d$ separate passes. In general, a Jacobian diagonal has the **same cost as the full jacobian**. 

Not sure I understand this. But apparently, one could use **HollowNets** to efficiently compute dimension-wise derivatives of order $k$.

**Source**: Ricky Chen [page](https://www.cs.toronto.edu/~rtqichen/)

---

## Interesting

### Continuous Normalizing flows

**FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models** - Grathwohl & Chen et. al. (2018) - [arxiv](https://arxiv.org/abs/1810.01367)

**Stochastic Normalizing Flows** - Hodgkinson et. al. (2020) - [arxiv](https://arxiv.org/abs/2002.09547)
