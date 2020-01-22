# Information Theory Measures Estimators

Information theory measures are potentially very useful in many applications ranging from Earth science to neuroscience. Measures like entropy enable us to summarise the uncertainty of a dataset, total correlation enable us to summarise the redundancy of the data, and mutual information enable to calculate the similarities between two or more datasets. However, all of these measures require us to estimate the probabilidy density function (PDF) and thus we need to create information theory estimators (ITEs). Each measure needs a probability distribution function $p(X)$ and this is a difficult task especially with large high-dimensional datasets. In this post, I will be outlining a few key methods that are used a lot in the literature as well ones that I am familiar with.

As a side note, I would like to stress that most conventional IT estimators work on 1D or 2D variables. Rarely do we see estimators designed to work for datasets with a high number of samples and/or a high number of dimensions. The estimator that our lab uses (RBIG) is designed exactly for that.



---

## Convential Estimators

Now lets consider the conventional estimators of the measures considered in this work: H, T, I and DKL.

While plenty of methods focus on the estimation of the above magnitudes for one dimensional or two-dimensional variables [?], there are few methods that deal with variables of arbitrary dimensions. Here we focus our comparative in the relatively wide family of methods present in this recent toolbox [[1]](1) which address the general multivariate case. Specifically, the family of methods in [7] is based on the following literature [8]–[18], which tackles the estimation problems according to different strategies.

[1]: blah	"ITE ToolBox Paper"



---

## Gaussian Assumption

The simplest ITM estimator is to assume that the data distributions are Gaussian. With this Gaussian assumption, the aforementioned ITMs are straightforward to calculate as the close-form solution just involves the approximated covariance matrix of the distribution(s). This assumption is very common to use for many scientific fields as datasets with a high number of samples tend to have a mean that is Gaussian distributed. However, this is not true for the tails of the distribution. In addition, the Gaussian distribution is in the exponential family of distributions which have attractive properties of mathematical convenience. The Sharma-Mittal entropy yields an analytical expression for the entropy of a Gaussian distribution \cite{Nielsen_2011} which also includes the derivative of the log-Normalizer of the distribution which acts as a corrective term yielding better estimates of the distribution. Omitting the error due to noise, the experiments will in the following section will illustrate that the Gaussian assumption should perform well on simple distributions but should perform worse for distributions which are characterized by their tails like the T-Student or the Pareto distribution.

---

## k-Nearest Neighbours Estimator

The next family of methods which are very commonly used are those with binning strategies. These binning methods include algorithms like the ridged histogram estimation, the smooth kernel density estimator (KDE), and the adaptive k-Nearest Neighbours (kNN) estimator \cite{Goria05}. These methods are non-parametric which allows them to be more flexible and should be able to capture more complex distributions. However, these methods are model-based and hence are sensitive to the parameters chosen and there is no intuitive method to choose the appropriate parameters. It all depends on the data given. We chose the most robust method, the kNN estimator, for our experiments which is adaptive in nature due to the neighbour structure from a distance matrix.  The kNN algorithms is known to have problems at scale with added problems of high dimensionality so we also included the scaled version which uses partition trees \cite{Stowell09} which sacrifices accuracy for speed.

---

## Von Mises Expansion


### Algorithms

##### KnnK

This is the most common method to estimate multivariate entropy used by the many different communities is the k-nearestest neighbours (kNN) estimator which was originally proposed by Kozachenko & Leonenko (1987); inspired by Dobrushin (1958). 



* Asymptotically unbiased
* Weakly consistent
* Non-parametric strategy which can heavily depend on the number of neighbours $k$ chosen. There has been some work done on this in (**citation**) but this issue is still an open problem for the community.

This method works by taking the distance from the $k^{th}$ sample fromDeviations of this method can be seen in common machine learning packages such as *sklearn* to estimate mutual information as well in large scale global causal graphs (**E**: Jakob's paper). 



 This uses the KNN method to estimate the entropy. From what I  understand, it's the simplest method that may have some issues at higher dimensions and large number of samples (normal with KNN estimators). In relation to the other standard methods of density estimation, it is the most robust in higher dimensions due to its adaptive-like binning.


* A new class of random vector entropy estimators and its applications in testing statistical hypotheses - Goria et. al. (2005) - [Paper](https://www.tandfonline.com/doi/full/10.1080/104852504200026815)
* Nearest neighbor estimates of entropy - Singh et. al. (2003) - [paper]()
* A statistical estimate for the entropy of a random vector - Kozachenko et. al. (1987) - [paper]()

##### KDP

This is the logical progression from KnnK. It uses KD partitioning trees (KDTree) algorithm to speed up the calculations I presume.

* Fast multidimensional entropy estimation by k-d partitioning - Stowell & Plumbley (2009) - [Paper]()

##### expF 

This is the close-form expression for the Sharma-Mittal entropy calculation for expontial families. This estimates Y using the maximum likelihood estimation and then uses the analytical formula for the exponential family.

* A closed-form expression for the Sharma-Mittal entropy of exponential families - Nielsen & Nock (2012) - [Paper]()

##### vME

This estimates the Shannon differential entropy (H) using the von Mises expansion. 

* Nonparametric von Mises estimators for entropies, divergences and mutual informations - Kandasamy et. al. (2015) - [Paper]()

##### Ensemble

Estimates the entropy from the average entropy estimations on groups of samples


This is a simple implementation with the freedom to choose the estimator `estimate_H`.

```python
# split into groups
for igroup in batches:
    H += estimate_H(igroup)
    
H /= len(batches)
```

* High-dimensional mutual information estimation for image registration - Kybic (2004) - [Paper]()


#### Potential New Experiments

#### Voronoi

Estimates Shannon entropy using Voronoi regions. Apparently it is good for multi-dimensional densities.

* A new class of entropy estimators for multi-dimensional densities - Miller (2003) - [Paper]()
