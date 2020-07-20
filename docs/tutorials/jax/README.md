# Jax

A new library that I think will take on the machine learning world and be very competitive with deep learning architectures. It's fast, runs on GPUs and you can use the native numpy way of coding. Below are some helpful resources as well as some of my own hand-crafted tutorials.

---

## Learning Jax

If you want more information about Jax in general:

1. Jax - Scipy Lecture 2020 - Jake VanderPlas - [Video](https://www.youtube.com/watch?v=z-WSrQDXkuM&feature=emb_logo)
   > A really good introduction to Jax by one of the contributors. Really aimed at the general public but detailed enough so that you can get a really good idea about how to use it.
2. You Don't Know Jax - Colin Raffel - [Blog](https://colinraffel.com/blog/you-don-t-know-jax.html)
   > A very simple introduction in blog format. Goes through the basics and is very clear.
3. Getting started with Hax (MLPs, CNNs, & RNNs) - Robert Tjarko Lange - [Video](https://roberttlange.github.io/posts/2020/03/blog-post-10/)
   > A nice tutorial but dives a bit deeper into some of the deep learning models you'll typically find. Implements everything from scratch and highlights the jax features in the process. Then proceeds to use the built-in `stax` library and refactors some parts. Probably the most complete advanced tutorial you'll find right now.
4. From PyTorch to Jax: towards neural net frameworks that purify stateful code - Sabrina J. Mielke - [blog](https://sjmielke.com/jax-purify.htm)
   > This tutorial goes from scratch and then really gets into the nitty gritty aspects that allow you to really customize your code to ensure that you follow the functional paradigm. It even shows you how to get 'classes' while still respecting the jax restrictions. From a philosophy standpoint, I think it's just awesome. And from a practical standpoint, it basically goes from a simple Jax function to something you'll see in the [Haiku]() library (a fully formed jax library for neural networks). Just awesome.
5. Accelerated ML Research via Composable Function Transformations in Python - [neurips 2019](https://slideslive.com/38922046/program-transformations-for-ml-3)
   > A tutorial about Jax but mainly from a programming research standpoint.
6. Taylor-Mode AD for Higher-Order Derivatives in Jax - [neurips 2019](https://slideslive.com/38922047/program-transformations-for-ml-4)
   > Very similar to the above tutorial.

---

## My Tutorials

I plan to use Jax and get accustomed to functional programming. I quite like it and it's a different way of doing things. Jax itself is super interesting and I really like the bells and whistles that it has to offer. Below are a few tutorials that I did to familiarize myself with it including `vmap`, `jit`, and `classes`. I'll be looking at `grad` in the future, in particular focusing on `kernels`.

* [vmap](vmap.md)
  > Automatic handling of batch dimensions (samples) so that you can write your code in vector format.
* [Jit Compilation](jit.md)
  > Jit compilation. Making your code fast with some restrictions.
* [Classes](classes.md)
  > Just a small example of the change in philosophy. Most deep learning libraries have you write things in terms of classes. But jax is purely functional (unless you use a dedicated library like [haiku](https://github.com/deepmind/dm-haiku)) so you need to change your coding style.

---

## Cool Libraries I follow

This is just a rolling list of really cool libraries that I follow. Sometimes I actively use it and sometimes I just like to browse code and get good ideas.

* [Flax]()
  > This should probably be in your repertoire of libraries. A really good and simple library for neural networks from Google itself. Strictly functional and super popular.
* [Haiku]()
  > Another very popular deep learning library built on top of Jax. This one gives the illusion of PyTorch/TensorFlow because the modules look very class oriented. But it still follows Jax protocol. Very neat how they managed to do that. To see it from scratch, see the [above](https://sjmielke.com/jax-purify.htm) tutorial.
* [Elegy]()
  > A new library based on Jax and Haiku which has a similar style to keras. Still very new but it has potential. I find it interesting because the natural progression from Jax+Haiku is something similar to keras. I'm glad someone took up that mantle.
* [Numpyro]() | [Paper](https://openreview.net/forum?id=H1g1niFhIB)
  > A probabilistic framework which focuses on mcmc sampling schemes (e.g. HMC/NUTS). It also has variational inference procedures.
* [Kalman Jax]()
  > This library is used for Markov GPs for time series. But they have a lot of little GP nuggets. Especially approximate inference algorithms, e.g. extended EP, statistically linearized EP, extended EP, etc.
* [NuX]()
  > Normalizing Flows using jax
* [jax-flows]()
  > Normalizing Flows using Jax.
* [Jax Cosmo]()
  > Applied to astrophysics but they have some nice routines that are not found in the main jax library (e.g. quad and interp)
