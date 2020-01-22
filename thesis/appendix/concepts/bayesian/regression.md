In typical regression problems we have some data $\mathcal{D}$ which consists of some input-output pairs $X,y$. We wish to find a function $f(\cdot)$ that maps the data $X$ to $y$. We also assume that there is some noise in the outputs $\epsilon_y$. We can also have noise on the inputs $X$ but we will discuss that at a later time. So concretely, we have:
$$
\begin{aligned}
y &= w \: x + \epsilon_y \\
\epsilon_y &\sim \mathcal{N}(0, \sigma_y^2)
\end{aligned}
$$
Let's demonstrate this by generating N data points from the true distribution.

<p float='center'> 
   <img src="pics/lin_reg_sample.png" width="375" />
  <img src="pics/lin_reg_weight.png" width="375" />
</p>

As seen from the figure above, the points that we generated line somewhere along the true line. Of course, we are privvy to see the true like but an algorithm might have trouble with such few points. In addition, we can see the weight space is quite large as well. One thing we can do is maximize the likelihood that $y$ comes from some normal distribution $\mathcal{N}$ with some mean $\mu$ and standard deviation $\sigma^2$. 
$$
\mathcal{F} = \underset{w}{\text{max}} \log \mathcal{N} (y_i | w \: x_i, \sigma^2) 
$$
Ultimately, after some optimization scheme, we would find a set of parameters for our problem, $\theta=(w, \sigma^2)$.

**Note**: this is actually equivalent to minimizing the mean-squared error (MSE) or minimizing the Kullback-Leibler Divergence. 
