# Information Theory Measures

## References

* Lecture Notes I - [PDF](http://www.ece.tufts.edu/ee/194NIT/lect01.pdf)
* Video Introduction - [Youtube](https://www.youtube.com/watch?v=ErfnhcEV1O8)

---
## Entropy

### One Random Variable

If we have a discrete random variable X with p.m.f. $p_x(x)$, the entropy is:

$$H(X) = - \sum_x p(x) \log p(x) = - \mathbb{E} \left[ \log(p(x)) \right]$$

* This measures the expected uncertainty in $X$.
* The entropy is basically how much information we learn on average from one instance of the r.v. $X$.

### Two Random Variables

If we have two random variables $X, Y$ jointly distributed according to the p.m.f. $p(x,y)$, we can come up with two more quantities for entropy.

#### Joint Entropy

This is given by:

$$H(X,Y) = \sum_{x,y} p(x,y) \log p(x,y) = - \mathbb{E} \left[ \log(p(x,y)) \right]$$

**Definition**: how much uncertainty we have between two r.v.s $X,Y$.

#### Conditional Entropy

This is given by:

$$H(X|Y) = \sum_{x,y} p(x,y) \log p(x|y) =  - \mathbb{E} \left[ \log ( p(x|y)) \right]$$

**Definition**: how much uncertainty remains about the r.v. $X$ when we know the value of $Y$.

## Properties of Entropic Quantities

* **Non-Negativity**: $H(X) \geq 0$, unless $X$ is deterministic (i.e. no randomness).
* **Chain Rule**: You can decompose the joint entropy measure:

    $$
    H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n}H(X_i | X^{i-1})
    $$ 
    
    where $X^{i-1} = \{ X_1, X_2, \ldots, X_{i-1} \}$. So the result is:

    $$H(X,Y) = H(X|Y) + H(Y) = H(Y|X) + H(X)$$

* **Monotonicity**: Conditioning always reduces entropy. *Information never hurts*.

    $$H(X|Y) \leq H(X)$$

---
## Mutual Information

**Definition**: The mutual information (MI) between two discreet r.v.s $X,Y$ jointly distributed according to $p(x,y)$ is given by:

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

**Sources**:
* [Scholarpedia](http://www.scholarpedia.org/article/Mutual_information)

## Conditional Mutual Information

**Definition**: Let $X,Y,Z$ be jointly distributed according to some p.m.f. $p(x,y,z)$. The conditional mutual information $X,Y$ given $Z$ is:

$$I(X;Y|Z) = - \sum_{x,y,z} p(x,y,z) \log \frac{p(x,y|z)}{p(x|z)p(y|z)}$$

$$I(X;Y|Z) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y|Z) = H(X) + H(Y) - H(X,Y)$$


---

## Total Correlation (Multi-Information)

In general, the formula for Total Correlation (TC) between two random variables is as follows:

$$TC(X,Y) = H(X) + H(Y) - H(X,Y)$$

**Note**: This is the same as the equation for mutual information between two random variables, $I(X;Y)=H(X)+H(Y)-H(X,Y)$. This makes sense because for a Venn Diagram between two r.v.s will only have one part that intersects. This is different for the multivariate case where the number of r.v.s is greater than 2.

Let's have $D$ random variables for $X = \{ X_1, X_2, \ldots, X_D\}$. The TC is:

$$TC(X) = \sum_{d=1}^{D}H(X_d) - H(X_1, X_2, \ldots, X_D)$$

In this case, $D$ can be a feature for $X$.

Now, let's say we would like to get the **difference in total correlation** between two random variables, $\Delta$TC.

$$\Delta\text{TC}(X,Y) =  \text{TC}(X) - \text{TC}(Y)$$

$$\Delta\text{TC}(X,Y) =  \sum_{d=1}^{D}H(X_d) - \sum_{d=1}^{D} H(Y_d) - H(X) + H(Y)$$

**Note**: There is a special case in [RBIG](https://github.com/jejjohnson/rbig) where the two random variables are simply rotations of one another. So each feature will have a difference in entropy but the total overall dataset will not. So our function would be reduced to: $\Delta\text{TC}(X,Y) =  \sum_{d=1}^{D}H(X_d) - \sum_{d=1}^{D} H(Y_d)$ which is overall much easier to solve.

---

## Cross Entropy (Log-Loss Function)

Let $P(\cdot)$ be the true distribution and $Q(\cdot)$ be the predicted distribution. We can define the cross entropy as:

$$H(P, Q) = - \sum_{i}p_i \log_2 (q_i)$$

This can be thought of the measure in information length.

**Note**: The original cross-entropy uses $\log_2(\cdot)$ but in a supervised setting, we can use $\log_{10}$ because if we use log rules, we get the following relation $\log_2(\cdot) = \frac{\log_{10}(\cdot)}{\log_{10}(2)}$.

## Kullback-Leibler Divergence (KL)

Furthermore, the KL divergence is the difference between the cross-entropy and the entropy.

$$D_{KL}(P||Q) = H(P, Q) - H(P)$$

So this is how far away our predictions are from our actual distribution.



---
## Supplementary Material

### GPs and IT







---
## References

#### Gaussian Processes and Information Theory


#### Information Theory


* Information Theory Tutorial: The Manifold Things Information Measures - [YouTube](https://www.youtube.com/watch?v=34mONTTxoTE)
* [On Measures of Entropy and Information](http://threeplusone.com/on_information.pdf) - 
* [Understanding Interdependency Through Complex Information Sharing](https://pdfs.semanticscholar.org/de0b/e2001efc6590bf28f895bc4c42231c6101da.pdf) - Rosas et. al. (2016)
* The Information Bottleneck of Deep Learning - [Youtube](https://www.youtube.com/watch?v=XL07WEc2TRI)


---
## Software

### Python Packages

* Discrete Information Theory (**dit**) - [Github](https://github.com/dit/dit) | [Docs](http://dit.readthedocs.io/en/latest/index.html)
* Python Information Theory Measures (****) - [Github](https://github.com/pafoster/pyitlib/) | [Docs](https://pafoster.github.io/pyitlib/)
* Parallelized Mutual Information Measures - [blog](http://danielhomola.com/2016/01/31/mifs-parallelized-mutual-information-based-feature-selection-module/) | [Github](https://github.com/danielhomola/mifs)

### Implementations

* Mutual Information Calculation with Numpy - [stackoverflow](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)

---
## Unexplored Stuff

These are my extra notes from resources I have found.

---
---
### GPs, Entropy, Residuals

> Gustau: Easy to compute stuff out of a GP (which is a joint multivariate Gaussian with covariance K) would be:
> 1) Differential) entropy from the GP:
> 
> H(X) = 0.5* log( (2*\pi*e)^n * det(K_x) ),
> 
> where K_x is the kernel matrix (a covariance after all in feature space).
> https://en.wikipedia.org/wiki/Differential_entropy#Properties_of_differential_entropy
> http://www.gaussianprocess.org/gpml/chapters/RW.pdf  (see e.g. A.5, eq. A.20, etc)
> 
> 2) And then I remembered that the LS error estimate could be bounded, and since a GP is after all LS regression in feature space, maybe we could check if the formula is right:
> MSE = E[(Y-\hat Y)^2] >= 1/(2*pi*e) * exp(H(Y|X))
> https://en.wikipedia.org/wiki/Conditional_entropy
> that is, MSE obtained with the GP is lower bounded by that conditional entropy estimate from RBIG.
> 
> Some building blocks to start with connecting dots... :)
> 
> Jesus
> Intuition said that "the bigger the conditional entropy, the bigger the residual uncertainty is -> the bigger the MSE should be (no matter the model)"
> Cool wiki formula: MSE >= exp(conditional)


$$H(X_y|X_A)=\frac{1}{2} \log \left(\nu_{X_y|X_A}^2 \right) + \frac{1}{2} \left( \log(2\pi) + 1 \right)$$

where:

* Conditional Variance: $\nu^2_{y|A}=K_{yy}- \Sigma_{y|A}\Sigma_{AA}^{-1}\Sigma_{Ay}$


**Unread Stuff**:

* Conditional Entropy in the Context of GPs - [Stack](https://stats.stackexchange.com/questions/388761/conditional-entropy-in-the-context-of-gaussian-processes)
* Entripy of a GP (Log(det(Cov))) - [stack](https://stats.stackexchange.com/questions/377794/entropy-of-a-gaussian-process-logdeterminantcovariancematrix)
* Get full covariance matrix and find its entropy - [stack](https://stackoverflow.com/questions/53345624/gpflow-get-the-full-covariance-matrix-and-find-its-entropy)


### GPs and Causality

* Intro to Causal Inference with GPs - [blog I](https://mindcodec.ai/2018/10/22/an-introduction-to-causal-inference-with-gaussian-processes-part-i/) | [blog II](https://mindcodec.ai/2018/12/09/an-introduction-to-granger-causality-using-gaussian-processes-part-ii/)
