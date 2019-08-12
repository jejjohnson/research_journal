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


#### Code - Step-by-Step

1. Obtain all of the possible occurrences of the outcomes. 
   ```python
   values, counts = np.unique(labels, return_counts=True)
   ```

2. Normalize the occurrences to obtain a probability distribution
   ```python
   counts /= counts.sum()
   ```

3. Calculate the entropy using the formula above
   ```python
   H = - (counts * np.log(counts, 2)).sum()
   ```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.

2. Use Scipy Function
   ```python
   H = entropy(counts, base=base)
   ```



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

