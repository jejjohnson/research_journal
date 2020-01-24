# Regression

* Bayesian Regression
* Inference

---

## Bayesian Regression


### Model

In typical regression problems we have some data $\mathcal{D}$ which consists of some input-output pairs $X,y$. We wish to find a function $f(\cdot)$ that maps the data $X$ to $y$. We also assume that there is some noise in the outputs $\epsilon_y$. We can also have noise on the inputs $X$ but we will discuss that at a later time. So concretely, we have:
$$
\begin{aligned}
y &= w \: x + \epsilon_y \\
\epsilon_y &\sim \mathcal{N}(0, \sigma_y^2)
\end{aligned}
$$
Let's demonstrate this by generating N data points from the true distribution.

<p float='center'> 
  <img src="pics/lin_reg_sample.png" width="300" />
  <img src="pics/lin_reg_weight.png" width="300" />
</p>

As seen from the figure above, the points that we generated line somewhere along the true line. Of course, we are privvy to see the true like but an algorithm might have trouble with such few points. In addition, we can see the weight space is quite large as well. One thing we can do is maximize the likelihood that $y$ comes from some normal distribution $\mathcal{N}$ with some mean $\mu$ and standard deviation $\sigma^2$. 
$$
\mathcal{F} = \underset{w}{\text{max}} \log \mathcal{N} (y_i | w \: x_i, \sigma^2) 
$$
So we will use the mean squared error (MSE) error as a loss function for our problem as maximizing the likelihood is equivalent to minimizing the MSE.

<details>
  <summary>
      <font color="red">
      Proof: 
      </font>
      max MLE = min MSE
  </summary>

The likelihood of our model is:

$$\log p(y|\mathbf{X,w}) = \sum_{i=1}^N \log p(y_i|x_i,\theta)$$

And for simplicity, we assume the noise $\epsilon$ comes from a Gaussian distribution and that it is constant. So we can rewrite our likelihood as

$$\log p(y|\mathbf{X,w}) = \sum_{i=1}^N \log \mathcal{N}(y_i | \mathbf{x_i, w}, \sigma^2)$$

Plugging in the full formula for the Gaussian distribution with some simplifications gives us:

$$
\log p(y|\mathbf{X,w}) = 
\sum_{i=1}^N 
\log \frac{1}{\sqrt{2 \pi \sigma_e^2}} 
\exp\left( - \frac{(y_i - \mathbf{x_iw})^2}{2\sigma_e^2} \right)
$$

We can use the log rule $\log ab = \log a + \log b$ to rewrite this expression to separate the constant term from the exponential. Also, $\log e^x = x$.

$$
\log p(y|\mathbf{X,w}) =
- \frac{N}{2} \log 2 \pi \sigma_e^2 
- \sum_{i=1}^N \frac{(y_i - \mathbf{x_iw})^2}{2\sigma_e^2}
$$

So, the first term is constant so that we can ignore that in our loss function. We can do the same for the denominator for the second term. Let's simplify it to make our life easier.

$$
\log p(y|\mathbf{X,w}) =
- \sum_{i=1}^N (y_i - \mathbf{x_iw})^2
$$

So we want to maximize this quantity: in other words, I want to find the parameter $\mathbf{w}$ s.t. this equation is maximum.

$$
\mathbf{w}_{MLE} = \argmax_{\mathbf{w}} - \sum_{i=1}^N (y_i - \mathbf{x_iw})^2
$$

We can rewrite this expression because the maximum of a negative quantity is the same as minimizing a positive quantity.

$$
\mathbf{w}_{MLE} = \argmin_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{x_iw})^2
$$

This is the same as the MSE error expression; with the edition of a scalar value $1/N$.

$$
\begin{aligned}
\mathbf{w}_{MLE} &= \argmin_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{x_iw})^2 \\
&= \argmin_{\mathbf{w}} \text{MSE}
\end{aligned}
$$

**Note**: If we did not know $\sigma_y^2$ then we would have to optimize this as well. 

</details>

### Inference