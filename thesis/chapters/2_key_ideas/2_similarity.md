# Similarity


## What is it?

When making comparisons between objects, the simplest question we can ask ourselves is how 'similar' one object is to another. It's a simple question but it's very difficult to answer. Simalirity in everyday life is somewhat easy to grasp intuitively but it's not easy to convey specific instructions to a computer. A [1](1). For example, the saying, "it's like comparing apples to oranges" is usually said when you try to tell someone that something is not comparable. But actually, we can compare apples and oranges. We can compare the shape, color, composition and even how delicious we personally think they are. So let's consider the datacube structure that was mentioned above. How would we compare two variables $z_1$ and $z_2$.

### Trends

In this representation, we are essentially doing one type or processing and then

* A parallel coordinate visualization is practical byt only certain pairwise comparisons are possible.

If we look at just the temporal component, then we could just plot the time series at different points along the globe.

> Trends often do not expose similarity in an intuitive way.

---

## Constraints

### Orthogonal Transformations

### Linear Transformations

### Isotropic Scaling

### Multivariate

#### Curse of Dimensionality

---

## Classic Methods

#### Summary Statistics

* Boxplots allow us to sumarize these statistics



#### Taylor Diagram

>  A Taylor Diagram is a concise statistical summary of how well patterns match each other in  terms of their correlation, their root-mean-square difference and the  ratio of their variances.

With this diagram, we can simultaneously plot each of the summary statistics, e.g. standard deviation, root mean squared error (RMSE) and the R correlation coefficient. The original reference can be found here [[2]](2).
$$
\text{RMS}^2 = \sigma_x^2+\sigma_y^2-2\sigma_x\sigma_yR_{xy} \\
c^2 = a^2 + b^2 - 2ab \cos \theta
$$


**This is a well understood diagram in the Earth science and climate community. It is also easy to compute.**

### Correlation



* A scatterplot matrix can be impractical for many outputs.



**Example with Anscombes Quartet** why IT measures might be useful for correlation plots.

---

## HSIC

---

## Mutual Information

### Information Theory

* Mutual Information is the counterpart to using information theory methods.
* It requires an estimation step which may introduce additional uncertainties
* Extends nicely to different types of data (e.g. discrete, categorical, multivariate, multidimentional)
* Exposes non-linearities which may be difficult to see via (linear) correlations
* Kernel Approximations: Although there are some differences for different estimators, relative distances are consistent

#### A Primer

* Entropy - measure of information uncertainty of $X$
* Joint Entropy - uncertinaty of $X,Y$
* Conditional Entropy - uncertainty of $X$ given that I know $Y$
* Mutual Information - how much knowning $X$ reduces the uncertainty of $Y$
  * $I(X,Y)=$
* Normalized Mutual Information
  * $\tilde{I}(X,Y) = \frac{I(X,Y)}{\sqrt{H(X)H(Y)}}$

### Variation of Information

> A measure of distance in information theory space.

$$
VI(X,Y) = H(X|Y) + H(Y|X) \\
VI(X,Y) = H(X) + H(Y) -2I(X,Y)
$$

where:

* $VI(X,Y)=0$ Iff $X$ and $Y$ are the same
  * $H(X,Y)=H(X)=H(Y)=I(X,Y)$
* $VI(X,Y) < H(X,Y)$ If $X$ and $Y$ are different but dependent
  * $H(X,Y)<H(X) + H(Y)$
* $VI(X,Y)=H(X,Y)$ if $X$ and $Y$ are independent
  * $H(X,Y)=H(X) + H(Y)$
  * $I(X,Y)=0$





## Questions

1. Are there **correlations** across seasons or latitudes
2. Are there large descrepancies in the different outputs?



### Classes of Methods









---

## Resources



### Websites

* [Taylor Diagrams](https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/taylor-diagrams)
* 

### Papers

[1]:   "The Mutual Information Diagram for Uncertainty Visualization - Correa & Lindstrom (2012)"
[2]: 	"Summarizing multiple aspects of model performance in a single diagram"

