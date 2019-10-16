# Key Concepts


---
## Definitions

* Mathematics
* Geometry: Studying Shapees and Spaces
* Algebra: Studying Relationships
* Probability: Belief, Uncertainty
* Calculus: Mathematics of Change

---
---
## Gaussian Distributions


### Univariate Gaussian

$$\mathcal{P}(x|\mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left( -\frac{1}{2\sigma^2}(x - \mu)^2 \right)$$

### Multivariate Gaussian

$$\begin{aligned}
\mathcal{P}(x | \mu, \Sigma) &= \mathcal{N}(\mu, \Sigma) \\
&= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{\sqrt{\text{det}|\Sigma|}}\text{exp}\left( -\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu) \right)
\end{aligned}$$


### Joint Gaussian Distribution

$$\begin{aligned}\mathcal{P}(x, y) &= 
\mathcal{P}\left(\begin{bmatrix}x \\ y\end{bmatrix} \right) \\
&= \mathcal{N}\left( 
    \begin{bmatrix}
    a \\ b
    \end{bmatrix},
    \begin{bmatrix}
    A & B \\ B^{\top} & C
    \end{bmatrix} \right)
    \end{aligned}$$


### Marginal Distribution $\mathcal{P}(\cdot)$

We have the marginal distribution of $x$

$$\mathcal{P}(x) \sim \mathcal{N}(a, A)$$

and in integral form:

$\mathcal{P}(x) = \int_y \mathcal{P}(x,y)dy$

and we have the marginal distribution of $y$

$$\mathcal{P}(y) \sim \mathcal{N}(b, B)$$

### Conditional Distribution $\mathcal{P}(\cdot | \cdot)$

We have the conditional distribution of $x$  given $y$.

$$\mathcal{P}(x|y) \sim \mathcal{N}(\mu_{a|b}, \Sigma_{a|b})$$

where:

* $\mu_{a|b} = a + BC^{-1}(y-b)$
* $\Sigma_{a|b} = A - BC^{-1}B^T$

and we have the marginal distribution of $y$ given $x$

$$\mathcal{P}(y|x) \sim \mathcal{N}(\mu_{b|a}, \Sigma_{b|a})$$

where:

* $\mu_{b|a} = b + AC^{-1}(x-a)$
* $\Sigma_{b|a} = B - AC^{-1}A^T$

basically mirror opposites of each other. But this might be useful to know later when we deal with trying to find the marginal distributions of Gaussian process functions.

**Source**:

* Sampling from a Normal Distribution - [blog](https://juanitorduz.github.io/multivariate_normal/)
  > A really nice blog with nice plots of joint distributions.
* Two was to derive the conditional distributions - [stack](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution?noredirect=1&lq=1)

---
---
### [Identity Trick](https://www.shakirm.com/slides/MLSS2018-Madrid-ProbThinking.pdf)

I think the slides from the MLSS 2018 meeting is the only place that I have encountered anyone actually explicitly mentioning this Identity trick.

Given an integral problem: 

$$p(x) = \int p(x|z)p(z)dz$$ 

I can multiply by an arbitrary distribution which is equivalent to 1.

$$p(x)=\int p(x|z) p(z) \frac{q(z)}{q(z)}dz$$

Then I can regroup and reweight the integral

$$p(x) = \int p(x|z)\frac{p(z)}{q(z)}q(z)dz$$

This results in a different expectation that we initially had

$$p(x) = \underset{q(z)}{\mathbb{E}}\left[ p(x|z)\frac{p(z)}{q(z)} \right]$$

Examples:

* Importance Sampling
* Manipulate Stochastic gradients
* Derive Probability bounds
* RL for policy corrections


---


**Teaching Notes**

* [Transforming Density Functions](https://www.cl.cam.ac.uk/teaching/2003/Probability/prob11.pdf)
* [Transformation of RVs](https://faculty.math.illinois.edu/~r-ash/Stat/StatLec1-5.pdf)

---
## Change of Variables

This is after making some transformation function, we can find the probability of that function by simply multiplying

**Resources**:
* Youtube:
  * Professor Leonard - [How to Change Variables in Multiple Integrals (Jacobian)](https://www.youtube.com/watch?v=VVPu5fWssPg&t=3648s)
  * mrgonzalezWHS - [Change of Variables. Jacobian](https://www.youtube.com/watch?v=1TPFb1aKMvk&t=198s)
  * Kishore Kashyap - [Transformations I](https://www.youtube.com/watch?v=6iphG6-iTo4&t=213s) | [Transformations II](https://www.youtube.com/watch?v=WOdgojmlZSQ)
* MathInsight
  * [Double Integrals](https://mathinsight.org/double_integral_change_variables_introduction) | [Example](https://mathinsight.org/double_integral_change_variable_examples)
* Course
  * Cambridge
    * [Transforming Density Functions](https://www.cl.cam.ac.uk/teaching/0708/Probabilty/prob11.pdf)
    * [Transforming Bivariate Density Functions](https://www.cl.cam.ac.uk/teaching/0708/Probabilty/prob12.pdf)
  * Pauls Online Math Notes
    * [Change of Variables](http://tutorial.math.lamar.edu/Classes/CalcIII/ChangeOfVariables.aspx)

---
## Inverse Function Theorem


**Resources**:
* [Wiki](https://en.wikipedia.org/wiki/Inverse_function_theorem)
* YouTube
  * Prof Ghist Math - [Inverse Function Theorem](https://www.youtube.com/watch?v=LWk7hvY1Goc)
  * The Infinite Looper - [Inv Fun Theorem](https://www.youtube.com/watch?v=gS0TYC78lnw&t=186s)
  * Professor Leonard - [Fundamental Theorem of Calculus](https://www.youtube.com/watch?v=xjtEfS0vY2o&list=PLF797E961509B4EB5&index=29&t=0s) | [Derivatives of Inverse Functions](https://www.youtube.com/watch?v=HnsUNWNYZ28)

---
## KL Divergence


**Typical**:

$$\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x)\right] =
\int_\mathcal{X} q(\mathbf x) \log \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} d\mathbf x$$

**VI**:

$$\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x)\right] = -
\int_\mathcal{X} q(\mathbf x) \log \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} d\mathbf x$$

#### Positive and Reverse KL



* Density Ratio Estimation for KL Divergence Minimization between Implicit Distributions - [Blog](https://tiao.io/post/density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions/)
**Resources**:
* YouTube
  * Aurelien Geron - [Short Intro to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
  * Ben Lambert - [Through Secret Codes](https://www.youtube.com/watch?v=LJwtEaP2xKA)
  * Zhoubin - [Video](https://youtu.be/5KdWhDpeQvU)
    > A nice talk where he highlights the asymptotic conditions for MLE. The proof is sketched using the minimization of the KLD function.
* Blog
  * [Anna-Lena Popkes](https://github.com/zotroneneis/resources/blob/master/KL_divergence.ipynb)
  * [KLD Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
  * [KLD for ML](https://dibyaghosh.com/blog/probability/kldivergence.html)
  * [Reverse Vs Forward KL](http://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
  * [KL-Divergence as an Objective Function](https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/)
  * [NF Slides (MLE context)](https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-09-Introduction-to-Normalizing-Flows/slides.pdf)
  * [Edward](http://edwardlib.org/tutorials/klqp)
* Class Notes
  * Stanford - [MLE](https://web.stanford.edu/class/stats200/Lecture13.pdf) | [Consistency and Asymptotic Normality of the MLE](https://web.stanford.edu/class/stats200/Lecture14.pdf) | [Fisher Information, Cramer-Raw LB](https://web.stanford.edu/class/stats200/Lecture15.pdf) | [MLE Model Mispecification](https://web.stanford.edu/class/stats200/Lecture16.pdf)
* Code
  * [KLD py](https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7)
  * [NumPy/SciPy Recipes](https://www.researchgate.net/publication/278158089_NumPy_SciPy_Recipes_for_Data_Science_Computing_the_Kullback-Leibler_Divergence_between_Generalized_Gamma_Distributions)

---
## Derivative of an Inverse Function

* MathInsight - [Link](https://mathinsight.org/derivative_inverse_function)




---
### Solving Hard Integral Problems

[**Source**](https://www.cs.ubc.ca/~schmidtm/MLRG/GaussianProcesses.pdf) | Deisenroth - [Sampling](https://drive.google.com/file/d/1Ryb1zDzndnv1kOe8nT0Iu4OD6m0KC8ry/view)

* Numerical Integration (low dimension)
* Bayesian Quadrature
* Expectation Propagation
* Conjugate Priors (Gaussian Likelihood w/ GP Prior)
* Subset Methods (Nystrom)
* Fast Linear Algebra (Krylov, Fast Transforms, KD-Trees)
* Variational Methods (Laplace, Mean-Field, Expectation Propagation)
* Monte Carlo Methods (Gibbs, Metropolis-Hashings, Particle Filter)