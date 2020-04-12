---
title: Python Ecosystem
description: The Python EcoSystem
authors:
    - J. Emmanuel Johnson
path: docs/resources/python/software_stacks
source: python_stack.md
---

# Python Packages

<center>
<img src="resources/pics/scipy_stack.png" width="350">
</center>


---
## Specialized Stack

* [scikit-image]()
* [Statsmodels]()
* [xarray]()
* 


---
## Automatic Differentiation Stack

These programs are mainly focused on automatic differentiation (a.k.a. AutoGrad). Each package is backed by some big company (e.g. Google, Facebook, Microsoft or Amazon).There are many packages nowadays, each with their pros and cons, but I will recommend the most popular. In the beginning, static graphs (define first then run after) was the standard but nowadays the dynamic method (define/run as you go) is more standard due to its popularity amongst the research community. So the differences between many of the libraries are starting to converge.

Please go to [this webpage](resources/dl_software.md) for a more detailed overview of the SOTA deep learning packages in python.

---
## Kernel Methods

So I haven't found any designated kernel methods library (<span style="color:blue"> open market?!</span>) but there are packages that have kernel methods within them. There are many packages that have GPs.

* [scikit-learn]()
  > This library has some of the standard [kernel functions](https://scikit-learn.org/stable/modules/metrics.html) and kernel methods such as [KPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html), [SVMs](https://scikit-learn.org/stable/modules/svm.html), [KRR]() and some [kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) schemes such as the nystrom method and RFF. **Note**: they do not have the kernel approximation schemes actually integrated into the KRR algorithm (<span style="color:blue"> open market?!</span>). For that, you can see [my implementation](https://github.com/jejjohnson/kernellib).

--- 
## Visualization Stack


---
## Geospatial Processing Stack

* [Geopandas](http://geopandas.org/)
  > This would be the easiest package to use when we need to deal with shape 
    files. It also follows the pandas syntax closes with added plotting 
    capabilities. 
* [Rasterio]()
* [Shapely]()
* [rioxarray](https://corteva.github.io/rioxarray/html/index.html)
  > A useful package that allows you to couple geometries with xarray. You can 
    mask or reproject data. I like this package because it's simple and it 
    focuses on what it is good at and nothing else.

