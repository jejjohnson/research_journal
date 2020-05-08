# Gaussian Distribution


$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left( \frac{-x^2}{2\sigma^2} \right)
$$


## Entropy

The closed-form solution for entropy is:

$$
h(X) = \frac{1}{2}\log (2\pi e\sigma^2)
$$

??? details "**Derivation**"

    $$
    \begin{aligned}
    h(X) 
    &= - \int_\mathcal{X} f(X) \log f(X) dx \\
    &= - \int_\mathcal{X} f(X) \log \left( \frac{1}{\sqrt{2\pi}\sigma}\exp\left( \frac{-x^2}{2\sigma^2} \right) \right)dx \\
    &= - \int_\mathcal{X} f(X)
    \left[ -\frac{1}{2}\log (2\pi \sigma^2) - \frac{x^2}{2\sigma^2}\log e \right]dx \\
    &= \frac{1}{2} \log (2\pi\sigma^2) + \frac{\sigma^2}{2\sigma^2}\log e \\
    &= \frac{1}{2} \log (2\pi e \sigma^2)
    \end{aligned}
    $$

??? details "**Code**"

    === "From Scratch"

        ```python
        def entropy_gauss(sigma: float) -> float:
            return np.log(2 * np.pi * np.e * sigma**2)
        ```

    === "Refactored"

        ```python
        from scipy import stats

        H_g = stats.norm(scale=sigma).entropy()
        ```

