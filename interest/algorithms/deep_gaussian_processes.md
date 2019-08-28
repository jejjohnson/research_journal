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

[Paper](https://arxiv.org/abs/1610.04386) | [Video](https://vimeo.com/238221933) | [Code](https://github.com/mauriziofilippone/deep_gp_random_features)



---

## Code

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
