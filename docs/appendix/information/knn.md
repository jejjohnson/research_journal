# K-Nearest Neighbors Estimator



The full entropy expression:

$$
\hat{H}(\mathbf{X}) =
\psi(N) -
\psi(k) +
\log{c_d} +
\frac{d}{N}\sum_{i=1}^{N}
\log{\epsilon(i)}
$$

where:
* $\psi$ - the digamma function.
* $c_d=\frac{\pi^{\frac{d}{2}}}{\Gamma(1+\frac{d}{2})}$
* $\Gamma$ - is the gamma function
* $\epsilon(i)$ is the distance to the $i^{th}$ sample to its $k^{th}$ neighbour.