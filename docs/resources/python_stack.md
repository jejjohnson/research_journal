# Python Packages

<center>
<img src="resources/pics/scipy_stack.png" width="350">
</center>


---

- [Python Packages](#python-packages)
  - [Standard Stack](#standard-stack)
  - [Specialized Stack](#specialized-stack)
  - [Automatic Differentiation Stack](#automatic-differentiation-stack)
    - [Deep Learning](#deep-learning)
  - [Kernel Methods](#kernel-methods)
  - [Gaussian Processes](#gaussian-processes)
  - [Visualization Stack](#visualization-stack)
  - [Geospatial Processing Stack](#geospatial-processing-stack)

---
## Standard Stack

* [Numpy]()
  * [Intro to Numpy](https://www.youtube.com/watch?v=ZB7BZMhfPgk) | [Visual Guide](https://jalammar.github.io/visual-numpy/) | [From Python to Numpy](http://www.labri.fr/perso/nrougier/from-python-to-numpy/) | [100 Exercises](https://github.com/rougier/numpy-100) | [Broadcasting](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) | Einsum - [I](https://rockt.github.io/2018/04/30/einsum) | [II](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)
* [Scipy]()
* [Scikit-Learn]()
  > This **is the** de facto library for machine learning. It will have all of the standard algorithms for machine learning. They also have a lot of utilities that can be useful when preprocessing or training your algorithms. The API (the famous `.fit()`, `.predict()`, and `.transform()`) is great and has been adopted in many other machine learning packages. I highly recommend this.
* [Pandas]()
* [Matplotlib]()



---
## Specialized Stack

* [scikit-image]()
* [Statsmodels]()
* [xarray]()
* 


---
## Automatic Differentiation Stack

These programs are mainly focused on automatic differentiation (a.k.a. AutoGrad). Each package is backed by some big company (e.g. Google, Facebook, Microsoft or Amazon).There are many packages nowadays, each with their pros and cons, but I will recommend the most popular. In the beginning, static graphs (define first then run after) was the standard but nowadays the dynamic method (define/run as you go) is more standard due to its popularity amongst the research community. So the differences between many of the libraries are starting to converge.


* [TensorFlow](https://www.tensorflow.org/)
  > Backed by **Google** most widely known package and started making autograd super popular. It has a bit of a learning curve but through the years they have made it more and more manageable. There is now [Tensorflow 2.0](https://www.tensorflow.org/beta/) which will make it much easier to prototype with. If you want to be on the cutting edge of production then I would recommend using this software. It's also the oldest and most popular so they have a large [*models hub*](https://www.tensorflow.org/resources/models-datasets). 

* [PyTorch](https://pytorch.org/)
  > Backed by **Facebook**, this is the second most widely known package. It became super famous because of the dynamic graphs which makes it much easier to prototype and debug. No need to predefine graphs, nor keep track of all of the sessions. It also has its own [*stack of software*](https://pytorch.org/ecosystem) and [*models hub*](https://pytorch.org/hub). Most researchers are starting to use this software these days because it's more intuitive to the python community. But Tensorflow is slowly catching up so beware.
* [JAX]()
* 

### Deep Learning

This is slightly different than


* [Keras](https://keras.io/)
  > We also can't mention Tensorflow without keras as it is a wrapper that makes deeplearning super easy. It has a similar API to sklearn so most people can easily get involved with Deep learning.

---
## Kernel Methods

So I haven't found any designated kernel methods library (<span style="color:blue"> open market?!</span>) but there are packages that have kernel methods within them. There are many packages that have GPs.

* [scikit-learn]()
  > This library has some of the standard [kernel functions](https://scikit-learn.org/stable/modules/metrics.html) and kernel methods such as [KPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html), [SVMs](https://scikit-learn.org/stable/modules/svm.html), [KRR]() and some [kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) schemes such as the nystrom method and RFF. **Note**: they do not have the kernel approximation schemes actually integrated into the KRR algorithm (<span style="color:blue"> open market?!</span>). For that, you can see [my implementation](https://github.com/jejjohnson/kernellib).

---
## Gaussian Processes

* [scikit-learn/GaussianProcesses](https://scikit-learn.org/stable/modules/gaussian_process.html)
  > This is the easiest and quickest GP library to get started. It's from sklearn so most people will be able to run it. It isn't cutting edge and it doesn't scale well due to it not hosting any of the sparse methods (<span style="color:blue"> open market?!</span>). But for getting started, it's a good start.
* [GPy](https://sheffieldml.github.io/GPy/)
  > This is the most comprehensive python package for GPs. It has almost all of the latest algorithms from the leading institution of GPs lead by Neil Lawrence. Unfortunately the library is not updated very regularly so I would strongly suggest using the GPflow or GPytorch package. It's also a bit difficult to really understand the code but the examples are plentiful from a research and exploration perspective. Most likely they'll have the corner case using GPs. As a reference to some of the fundamental and quirky GP algorithms, check out GPy.
* [GPflow](https://www.gpflow.org/)
* [GPyTorch](https://gpytorch.ai/)
* [Pyro/contrib.gp](http://pyro.ai/examples/gp.html#)
* [TensorFlow Probability/GPR](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcessRegressionModel)


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
* [xESMF](https://xesmf.readthedocs.io/en/latest/why.html)
  > This is a nice regridding package which allows one to regrid your dataset
    in terms of another. Very useful for trying to get the same spatial 
    coordinates for two datasets.
