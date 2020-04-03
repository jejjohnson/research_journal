# Information Theory Measures

- [Summary](#summary)
- [Information](#information)
- [Entropy](#entropy)
- [Mutual Information](#mutual-information)
- [Total Correlation (Mutual Information)](#total-correlation-mutual-information)
- [Kullback-Leibler Divergence (KLD)](#kullback-leibler-divergence-kld)

## Summary

<!-- <img src="pics/rbig_it/Fig_1.png" alt="IT measures" width="300"> -->


<!-- <figure> -->
<center>
<img src="docs/pics/rbig_it/Fig_1.png" width="500">
</center>
<center>
<figurecaption>
<b>Caption</b>: Information Theory measures in a nutshell.
</figurecaption>
</center>
<!-- </figure> -->

## Information


## Entropy


## Mutual Information


## Total Correlation (Mutual Information)

This is a term that measures the statistical dependency of multi-variate sources using the common mutual-information measure.

$$
\begin{aligned}
I(\mathbf{x})
&= 
D_\text{KL} \left[ p(\mathbf{x}) || \prod_d p(\mathbf{x}_d) \right] \\
&= \sum_{d=1}^D H(x_d) - H(\mathbf{x})
\end{aligned}
$$

where $H(\mathbf{x})$ is the differential entropy of $\mathbf{x}$ and $H(x_d)$ represents the differential entropy of the $d^\text{th}$ component of $\mathbf{x}$. This is nicely summaries in equation 1 from ([Lyu & Simoncelli, 2008][1]).

?> Note: We find that $I$ in 2 dimensions is the same as mutual information.

We can decompose this measure into two parts representing second order and higher-order dependencies:

$$
\begin{aligned}
I(\mathbf{x}) 
&=
\underbrace{\sum_{d=1}^D \log{\Sigma_{dd}} - \log{|\Sigma|}}_{\text{2nd Order Dependencies}} \\
&-
\underbrace{D_\text{KL} \left[ p(\mathbf{x}) || \mathcal{G}_\theta (\mathbf{x}) \right] 
- 
\sum_{d=1}^D D_\text{KL} \left[ p(x_d) || \mathcal{G}_\theta (x_d) \right]}_{\text{high-order dependencies}}
\end{aligned}
$$

again, nicely summarized with equation 2 from ([Lyu & Simoncelli, 2008][1]).

**Sources**:
* Nonlinear Extraction of "Independent Components" of elliptically symmetric densities using radial Gaussianization - Lyu & Simoncelli - [PDF](https://www.cns.nyu.edu/pub/lcv/lyu08a.pdf)


[1]: https://www.cns.nyu.edu/pub/lcv/lyu08a.pdf "Nonlinear Extraction of 'Independent Components' of elliptically symmetric densities using radial Gaussianization - Lyu & Simoncelli - (2008)"

## Kullback-Leibler Divergence (KLD)