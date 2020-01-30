# Deep Learning Software for Python

### Core Packages


#### [TensorFlow](https://www.tensorflow.org/) (TF)

This is by far the most popular autograd library currently. Backed by Google (Alphabet, Inc), it is the go to python package for production and it is very popular for researchers as well. Recently (mid-2019) there was a huge update which made the python API much easier to use by having a tight keras integration and allowing for a more pythonic-style of coding.

**[TensorFlow Probability](https://www.tensorflow.org/probability)** (TFP)

As the name suggests, this is a probabilistic library that is built on top of TensorFlow. It has many distributions, priors, and inference methods. In addition, it uses the same `layers` as the `tf.keras` library with some `edward2` integration.

**[Edward2](https://github.com/google/edward2)** (Ed2)

While there is some integration of `edward2` into the TFP library, there are some stand alone functions in the original Ed2 library. Some more advanced `layers` such as Sparse Gaussian processes and Noise contrastive priors. It all works seemlessly with TF and TFP.

**[GPFlow](https://github.com/GPflow/GPflow)** (GPF)

This is a special library for SOTA GPs that's built on top of TF. It is the success to the original GP library, GPy but it has a much cleaner code base and a bit better documentation due to its use of autograd.

---

#### [PyTorch](https://pytorch.org/)

This is the most popular DL library for the machine learning community. Backed by Facebook, this is a rather new library that came out 2 years ago. It took a bit of time, but eventually people started using it more and more especially in a research setting. The reason is because it is very pythonic, it was designed by researchers and for researchers, and it keeps the design simple even sacrificing speed if needed. If you're starting out, most people will recommend you start with PyTorch.

**[Pyro](https://pyro.ai/)**

This is a popular Bayesian DL library that is built on top of PyTorch. Backed by Uber, you'll find a lot of different inference scheme. This library is a bit of a learning curve because their API. But, their documentation and tutorial section is great so if you take your time you should pick it up. Unfortunately I don't find too many papers with Pyro code examples, but when the Bayesian community gets bigger, I'm sure this will change.

**[GPyTorch](https://gpytorch.ai/)**

This is the most scalable GP library currently that's built on top of PyTorch. They mainly use Matrix-Vector-Multiplication strategies to scale exact GPs up to 1 million points using multiple GPUs. They recently just revamped their documentation as well with many examples. If you plan to put GPs in production, you should definitely check out this library. 


**[fastai](https://docs.fast.ai/)**

This is a DL library that was created to facilate people using some of the SOTA NN methods to solve problems. It was created by Jeremy Howard and has gained popularity due to its simplicity. It has all of the SOTA parameters by default and there are many options to tune your models as you see fit. In addition, it has a huge community with a very active [forum](https://forums.fast.ai/). I think it's a good first pass if you're looking to solve real problems and get your feet wet while not getting too hung up on the model building code. They also have 2 really good [courses](https://course18.fast.ai/) that teach you the mechanics of ML and DL while still giving you enough tools to make you a competitive.

---

**[JAX]()**

This is the successor for the popular [autograd](https://github.com/HIPS/autograd) package that is now backed by Google. It is basically `numpy` on steroids giving you access to an autograd function and it can be used on CPUs, GPUs and TPUs. The gradients are very intuitive as it is just using a `grad` function; recursively if you want high-order gradients. It's still fairly early in development but it's gaining popularity very fast and has shown to be [**very** competitive](https://github.com/dionhaefner/pyhpc-benchmarks). It's fast. Really fast. To really take advantage of this library, you will need to do a more functional-style of programming but there have been many benefits, e.g. using it for MCMC sampling schemes has become [popular](https://twitter.com/remilouf/status/1215740986195922944).

---

### Other Packages 

Although these are in the 'other' category, it doesn't mean that they are lower tier by any means. I just put them here because I'm not too familiar with them outside of the media. 

**[Chainer](https://chainer.org/)**

Preferred Networks (*Japanese Company*)

**[MXNet](https://mxnet.apache.org/)**

Amazon

**[Theano]()**

maintained by the PyMC3 developers.

**[PyMC3]()** & **[PyMC4]()**


**[CNTK]()**

Microsoft


