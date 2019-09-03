# Deep Gaussian Processes

This is my notebook for all of my resources on Deep Gaussian Processes.


---
## Algorithms

There are many family of algorithms

#### Original Deep GP

> This paper is the original method of Deep GPs. 

[Paper]() | [Code]()

#### Doubly Stochastic

> This paper uses stochastic gradient descent for training the Deep GP. I think this achieves the state-of-the-art results thus far. It also has the most implementations in the standard literature.

[Paper]() | [Code]()


#### Approximate Expectation Propagation

> This paper uses an approximate expectation method for the inference in Deep GPs.

[Paper]() | [Code]()


#### Random Fourier Features

> This implementation uses ideas from random fourier features in conjunction with Deep GPs.

* [Paper I](https://arxiv.org/abs/1610.04386) | [Paper II](https://pdfs.semanticscholar.org/bafa/7e2d586e7bfe77d9a55ac1cff4eb2f6ff292.pdf) |  [Video](https://vimeo.com/238221933) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features)
* [Lecture I]() | [Slides]() | 
* [Lecture (Maurizio)](https://www.youtube.com/watch?v=750fRY9-uq8&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=19&t=0s) | [Slides](http://www.eurecom.fr/~filippon/Talks/talk_deep_bayes_moscow_2019.pdf) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features/blob/master/code/dgp_rff.py)




---

## Code

---

## Insights

* Deep Gaussian Process Pathologies - [Paper](http://proceedings.mlr.press/v33/duvenaud14.pdf)
  > This paper shows how some of the kernel compositions give very bad estimates of the functions between layers; similar to how residual NN do much better.
* sd

---

## Cutting Edge

These are bleeding edge parts of Deep GPs that have peaked my interest.


#### Importance Weighted Sampling

[Paper](https://arxiv.org/abs/1905.05435) | [Code](https://github.com/hughsalimbeni/DGPs_with_IWVI) | [Video](https://slideslive.com/38917895/gaussian-processes) | [Poster](https://twitter.com/HSalimbeni/status/1137856997930483712/photo/1) | [Tweet]()

#### Generalized Variational Inference

> In this paper, the author looks at a generalized variational inference technique that can be applied to deep GPs.

[VI](https://arxiv.org/pdf/1904.02063.pdf) | [DeepGP](https://arxiv.org/pdf/1904.02303.pdf)




---

## Applications

#### Multi-Fidelity Modeling

> The users look at the case of multi-fidelity modeling using Deep GPs.

[Paper](https://arxiv.org/pdf/1903.07320.pdf) | [Code](https://github.com/amzn/emukit/tree/master/emukit/examples/multi_fidelity_dgp)
