# Kernels and Information Measures

This post will be based off of the paper from the following papers:

1. Measures of Entropy from Data Using Infinitely Divisible Kernels - Giraldo et. al. (2014)
2. Multivariate Extension of Matrix-based Renyi's $\alpha$-order Entropy Functional - Yu et. al. (2018)


## Kernel Matrices

$$\begin{aligned}
\hat{f}(x) &= \frac{1}{N} \sum_{i=1}^N K_\sigma (x, x_i) \\
K_\sigma(x_i, x_j) &= \frac{1}{(2\pi \sigma^2)^{d/2}}\exp\left( - \frac{||x-x_i||^2_2}{2\sigma^2} \right)  
\end{aligned}$$

## Entropy

$$H_2(x) = \log \int_\mathcal{X}f^2(x)\cdot dx$$

$$H_2(x) = - \log \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^N K_{\sqrt{2}\sigma}(x_i, x_j)$$

$$H_2(x) = - \log \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^N K_{\sqrt{2}\sigma}$$

**Note**: We have to use the convolution theorem for Gaussian functions. [Source](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjq6ZGs3P3jAhVRasAKHZ24BCcQFjABegQIDBAF&url=http%3A%2F%2Fwww.tina-vision.net%2Fdocs%2Fmemos%2F2003-003.pdf&usg=AOvVaw1SaNhee0xBCB561s0D8Jba) | 

#### Practically

$$\hat{H}_2(x) = - \log \frac{1}{N^2} \mathbf{1}_N^\top \mathbf{K}_x \mathbf{1}_N$$

where $\mathbf{1}_N \in \mathbf{R}^{1 \times N}$. The quantity $\mathbf{1}_N^\top \mathbf{K}_x \mathbf{1}_N$ is known as the *information potential*, $V$.

