# Loss Functions


Recall the change of variables formulation to calculate the probability:

$$p_\theta(x) = p_z(z) \; |\nabla_x \mathcal{G}_\theta(x)|$$

and we can also calculate the log probability like so:

$$\log p_\theta(x) = \log p_z(z) + \log |\nabla_x \mathcal{G}_\theta(x)|$$


where $z=\mathcal{G}_\theta(x)$.

---

### Negative Log-Likelihood

$$\mathbb{E}_\mathbf{x}\left[  p_\theta(x)\right] =
\mathbb{E}_x \left[ \log p_z(\mathcal{G}_\theta(x)) + \log |\nabla_x \mathcal{G}_\theta (x)| \right]$$

Empirically, this can be calculated by:

$$
\mathbb{E}_\mathbf{x}\left[  p_\theta(x)\right] =
\frac{1}{N} \sum_{i=1}^N \log p_z(\mathcal{G}_\theta(x_i)) + 
\frac{1}{N} \sum_{i=1}^N \log |\nabla_x \mathcal{G}_\theta (x_i)| 
$$

---

### Non-Gaussianity

$$J(p_y) = \mathbb{E}_x \left[  \log p_x(x) - \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| - \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right)\right]
$$

$$
\begin{aligned}
J(p_y) &=
\mathbb{E}_x \left[  \log p_x(x) \right] -
\mathbb{E}_x \left[  \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| \right] -
\mathbb{E}_x \left[  \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right) \right] \\
&=
\mathbb{E}_x \left[ \log p_z(\mathcal{G}_\theta(x)) 
\right] +
\mathbb{E}_x \left[ \log |\nabla_x \mathcal{G}_\theta (x)| \right] -
\mathbb{E}_x \left[  \log \left| \nabla_x \mathcal{G}_\theta(x)  \right| \right] -
\mathbb{E}_x \left[  \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right) \right]
\\
&=
\mathbb{E}_x \left[ \log p_z(\mathcal{G}_\theta(x)) 
\right] -
\mathbb{E}_x \left[  \log \mathcal{N}\left(\mathcal{G}_\theta(x)\right) \right]
\\
\end{aligned}
$$

which we can find empirically:

$$J(p_y) = 
\sum_{i=1}^N \log p_z(\mathcal{G}_\theta(x_i)) -
\sum_{i=1}^N \log \mathcal{N}\left(\mathcal{G}_\theta(x_i)\right)
$$

>! **Question**: What's the difference between the two equations? Perhaps part 1, you fit a Gaussian...

---

### Change in Total Correlation

---

### Change in Non-Gaussianity


$$\Delta J(p_y) = J(p_y) - J(p_x)$$

$$\Delta J(p_y) = \mathbb{E}_x \left[ \frac{1}{2} ||y||_2^2 - \log |\nabla_x \mathcal{G}_\theta (x)| - \frac{1}{2} ||x||_2^2 \right]$$

Empirically, we can calculate this by:

$$\Delta J(p_y) = \frac{1}{2} ||y||_2^2  - \frac{1}{2} ||x||_2^2 - \frac{1}{N}\sum_{i=1}^N 
\log |\nabla_x \mathcal{G}_\theta (x)|
$$