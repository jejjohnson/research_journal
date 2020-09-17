# Ecosystem

This is just a rolling list of really cool libraries that I follow. Sometimes I actively use it and sometimes I just like to browse code and get good ideas.

### Neural Networks

* [Flax]()
  > This should probably be in your repertoire of libraries. A really good and simple library for neural networks from Google itself. Strictly functional and super popular.
* [Haiku]()
  > Another very popular deep learning library built on top of Jax. This one gives the illusion of PyTorch/TensorFlow because the modules look very class oriented. But it still follows Jax protocol. Very neat how they managed to do that. To see it from scratch, see the [above](https://sjmielke.com/jax-purify.htm) tutorial.
* [Elegy]()
  > A new library based on Jax and Haiku which has a similar style to keras. Still very new but it has potential. I find it interesting because the natural progression from Jax+Haiku is something similar to keras. I'm glad someone took up that mantle.
* [Optax](https://github.com/deepmind/optax)
  > A library from deepmind that does gradient processing and optimization. Apparently it's based off of `jax.experimental.optix` which is being [phased out](https://twitter.com/SingularMattrix/status/1294733582041092096).


### Probabilistic Programming

* [Numpyro]() | [Paper](https://openreview.net/forum?id=H1g1niFhIB)
  > A probabilistic framework which focuses on mcmc sampling schemes (e.g. HMC/NUTS). It also has variational inference procedures.
* [mcx](https://github.com/rlouf/mcx)
  > A probabilistic programming library focused on sampling methods.
* [jaxns](https://github.com/Joshuaalbert/jaxns)
  > Nested sampling using Jax.

### Gaussian Processes

* [Kalman Jax]()
  > This library is used for Markov GPs for time series. But they have a lot of little GP nuggets. Especially approximate inference algorithms, e.g. extended EP, statistically linearized EP, extended EP, etc.


### Normalizing Flows


* [NuX]()
  > Normalizing Flows using jax
* [jax-flows]()
  > Normalizing Flows using Jax.
* [Jax Cosmo]()
  > Applied to astrophysics but they have some nice routines that are not found in the main jax library (e.g. quad and interp)
