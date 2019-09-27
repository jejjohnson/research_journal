# ITE Toolbox Algorithms


## Entropy 

### Algorithms

##### KnnK

This uses the KNN method to estimate the entropy. From what I understand, it's the simplest method that may have some issues at higher dimensions and large number of samples (normal with KNN estimators). 


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