# Software

- [GPy](#gpy)
    - [My Model Zoo](#my-model-zoo)
- [GPFlow](#gpflow)
- [Pyro](#pyro)
    - [My Model Zoo](#my-model-zoo-1)
- [GPyTorch](#gpytorch)
- [Summary](#summary)
  - [Algorithms Implemented](#algorithms-implemented)

---

Right now there are a few Python packages that do handle uncertain inputs. I try to focus on the libraries that offer the most built-infunctionality but also are the most extensible. 

!> **Note** If you want more information regarding the software, then please look at my software guide to GPs located [here](https://jejjohnson.github.io/gp_model_zoo/#/software). For more information specifically related to GPs for uncertain inputs, then keep reading.

**TLDR**:
* Like TensorFlow? Use GPFlow.
* Like PyTorch? Use Pyro.
* Lastest and greatest modern GPs? Use GPyTorch.

## [GPy](https://sheffieldml.github.io/GPy/)

This library has a lot of the original algorithms available regarding uncertain inputs. It will host the classics such as the sparse variational GP which offers an argument to specify the input uncertainty. However, the backend is the same as the Bayesian GPLVM. This library hasn't been updated in a while so I don't recommend users to use this regularly outside of small data problems.

#### My Model Zoo

* Exact GP Linearized - [github](https://github.com/jejjohnson/gp_model_zoo/tree/master/gpy)
* Sparse GP Linearized - [github](https://github.com/jejjohnson/gp_model_zoo/tree/master/gpy)
* Bayesian GPLVM - [github](https://github.com/jejjohnson/gp_model_zoo/tree/master/gpy)

## [GPFlow](https://github.com/GPflow/GPflow)

This library is the successor to GPy that is built on TensorFlow and TensorFlow Probability. It now features more or less most of the original algorithms from the GPy library but it is much cleaner because a lot of the gradients are handled automatically by TensorFlow. It is a good defacto library for working with GPs in the research setting.

## [Pyro](http://pyro.ai/)

This is a probabilistic library uses PyTorch as a backend. It features many inference algorithms such as Monte Carlo and Variational inference schemes. It has a barebones but really extensible GP library available. It is really easy to modify parameters and add prior distributions to whichever components is necessary. I find this library very easy to experiment with in my research.


#### My Model Zoo

* Sparse GP - [colab](https://colab.research.google.com/drive/1LIeYFaJPiguN2GDj5tYS_YZbg_GazKDa)
* Variational GP - [colab](https://colab.research.google.com/drive/15ViaPySxqicBp19AKvVRSuYjGHYzrWIP)
* Stochastic Variational GP - [colab](https://colab.research.google.com/drive/1WZOMte6OnSWmWnJLEAZoGp8z31HgGlvF)

## [GPyTorch](https://gpytorch.ai/)

This is a dedicated GP library with PyTorch as a backend. It has the most update features for using modern GPs. This also has some shared components with the Pyro library so it is now easier to modify parameters and add prior distributions. Right now, there is a bit of a learning curve if you want to use it outside of the use cases in the documentation. But, as they keep updating it, I'm sure utilizing it will get easier and easier; on par with Pyro or better. I recommend using this library when you want to move towards production or more extreme applications.

## Summary

### Algorithms Implemented

| **Package**               | **GPy** | **GPFlow** | **Pyro** | **GPyTorch** |
| ------------------------- | ------- | ---------- | -------- | ------------ |
| Linearized (Taylor)       | S       | S          | S        | S            |
| Exact Moment Matching GP  | ✗       | ✗          | ✗        | ✗            |
| Sparse Moment Matching GP | ✓       | ✗          | ✓        | ✗            |
| Uncertain Variational GP  | ✓       | S          | S        | S            |
| Bayesian GPLVM            | ✓       | ✓          | ✓        | S            |

**Key**

| Symbol | Status          |
| ------ | --------------- |
| **✓**  | **Implemented** |
| ✗      | Not Implemented |
| S      | Supported       |

