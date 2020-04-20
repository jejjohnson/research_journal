# Neural Networks with Uncertainty

- Author: J. Emmanuel Johnson
- Date: 1st October, 2019

## Synopsis

This document will be my notes on how one can classify different neural network architectures with regards to how they deal with uncertainty measures. My inspiration for this document comes from two factors:

1. My general interest in uncertainty (especially in the inputs which seems to be an unsolved problem)
2. The new tensorflow 2.0, tensorflow probability packages with really good blog posts (e.g. [here]() and [here]()) showing how one can use them to do probabilistic regression, 



---
## What is Uncertainty?

Before we talk about the types of neural networks that handle uncertainty, we first need to define some terms about uncertainty. There are three main types of uncertainty but they each 

* Aleatoric (Data)
	*  irreducible uncertainty
	*  when the output is inherently random - IWSDGP
* Epistemic (Model)
	* model/reducible uncertainty
	* when the output depends determininstically on the input, but there is uncertainty due to lack of observations - IWSDGP
	* 
* Distribution

Aleatoric uncertainty is the uncertainty we have in our data. We can break down the uncertainty for the Data into further categories: the inputs $X$ versus the outputs $Y$. We can further break down the types into homoscedastic, where we have continuous noise for the inputs and heteroscedastic, where we have uncertain elements per input.

#### Uncertainty in the Error Generalization

First we would like to define all of the sources of uncertainty more concretely. Let's say we have a model $y=f(x)+e$. For starters, we can decompose the generalization error term:

$$\mathcal{E}(\hat{f}) = \mathbb{E}\left[ l(f(x) + e, \hat{f}(x)) \right]$$

$$\mathcal{E}(\hat{f}) = \mathcal{E}(f) + \left( \mathcal{E}(\hat{f}) - \mathcal{E}(f^*) \right) + \left( \mathcal{E}(f^*) - \mathcal{E}(f) \right)$$

$$\mathcal{E}(\hat{f}) = 
    \underset{\text{Bayes Rate}}{\mathcal{E}_{y}} + \underset{\text{Estimation}}{\mathcal{E}_{x}} + \underset{\text{Approx. Error}}{\mathcal{E}_{f}}$$


where $\mathcal{E}_{y}$ is the best possible prediction we can achieve do to the noise $e$ thus it cannot be avoided; $\mathcal{E}_{x}$ is due to the finite-sample problem; and $\mathcal{E}_{f}$ is the model 'wrongness' (the fact that all models are wrong but some are useful). \textbf{Note:} as the number of samples decrease, then the model wrongness will increase. More samples will also allow us to decrease the estimation error. However, many times we are still certain of our uncertainty and we would like to propagate this knowledge through our ML model.

### Uncertainty Over Functions

In this section, we will look at the Bayesian treatment of uncertainty and will continue to define the terms aleatoric and epistemic uncertainty in the Bayesian language. Below we briefly outline the Bayesian model functionality in terms of Neural networks.

**Prior**:

$$p(w_{h,d}) = \mathcal{N}(w_{h,d} | 0, s^2)$$

where $W \in \mathbb{R}^{H \times D}$.

**Likelihood**

$$p(Y|X, W) = \prod_H \mathcal{N}(y_h | f^W(x_h), \sigma^2)$$

where $f^W(x) = W^T\phi(x)$, $\phi(x)$ is a N dimensional vector.

**Posterior**

$$P(W|X,Y) = \mathcal{N}(W| \mu, \Sigma)$$

where:

* $\mu = \Sigma \sigma^{-2}\Phi(X)^TY$
* $\Sigma = (\sigma^{-2} \Phi(X)^\top\Phi(X) + s^2\mathbf{I}_D)^{-1}$

**Predictive**

$$p(y^*|x^*, X, Y) = \mathcal{N}(y^*| \mu_*, \nu_{**}^2) $$

where:

* $\mu_* = \mu^T\phi(X^*)$
* $\nu_{**}^2 = \sigma^2 + \phi(x^*)^\top\Sigma \phi(x^*)$

Strictly speaking from the predictive uncertainty formulation above, uncertainty has two components: the variance from the likelihood term $\sigma^2$ and the variance from the posterior term $\nu_{**}^2$. 


#### Aleatoric Uncertainty, $\sigma^2$

This corresponds to there being uncertainty on the data itself. We assume that the measurements, $y$ we have some amount of uncertainty that is irreducible due to measurement error, e.g. observation/sensor noise or some additive noise component. A really good example of this is when you think of the dice player and the mean value and variance value of the rolls. No matter how many times you roll the dice, you won't ever reduce the uncertainty. If we can assume some model over this noise, e.g. Normally distributed, then we use maximum likelihood estimation (MLE) to find the parameter of this distribution. 

I want to point out that this term is often only assumed to be connected with $y$, the measurement error. They often assume that the $X$'s are clean and have no error. However, in many cases, I know especially in my field of Earth sciences, we have uncertainty in the $X$'s as well. This is important for error propagation which will lead to more credible uncertainty measurements. One way to handle this is to assume that the likelihood term $\sigma^2$ is not a constant but instead a function of $X$, $\sigma^2(x)$. This is one way to ensure that this variance estimate changes depending upon the value of X. Alternatively, we can also assume that $X$ is not really variable but instead a latent variable. In this formulation we assume that we only have access to some noisy observations $x_\mu$ and there is an additive noise component $\Sigma_x$ (which can be known or unknown depending on the application). In this instance, we need to propogate this uncertainty through each of the values within the dataset on top of the uncertain parameters. In the latent variable model community, they do look at this but I haven't seen too much work on this in the applied uncertainty community (i.e. people who have known uncertainties they would like to account for). I hope to change that one day...

#### Epistemic Uncertainty, $\nu_{**}^2$

The second term is the uncertainty over the function values before the noise corruption $\sigma^2$. In this instance, we find

---
## Overview of Architecture Types

So in terms of neural networks and uncertainty, I would say there are about 3 classes of neural networks ranging from no uncertainty to full uncertainty. They include:
* generic neural networks (NNs) which have no uncertainty
* Probabilistic Neural Networks (PNNs) which have uncertainty in the predictions
* Bayesian Neural Networks (BNNs) which have uncertainty on the weights as well

More concretely terms of what has distributions and what doesn't, we could classify them by where we put 

## Generic Neural Networks (NN)

This neural network is vanilla with no uncertainty at all. We have no probability assumptions about our data. We merely want some function $f$ that maps our data from $X \rightarrow Y$. 


**Summary**

* Very expressive architectures
* Training yields the best model parameters
* Cannot quantify the output uncertainty or confidence

## Probabilistic Neural Networks (PNN)

This class of neural networks are very cheap to produce. They basically attach a probability distribution on the final layer of the network. They don't have any probability distributions on and of the weights of the network. Another way to think of it is as a feature extractor that maps all of the data to a . Another big difference is the training procedure. Typically one would have to use a log-likelihood estimation but this isn't always necessary for PNN with simpler assumptions.

### Learning: Maximum Likelihood

Given some training data $D=(X_i, Y_i), i=1,2,\ldots,N$, we want to learn the parameters $W$ by maximizing the log likelihood i.e.

$$
\begin{aligned}
W^*&= \underset{W}{\text{argmax}} \log p(D|W) \\
\log p(D|W) &= \sum_{i=1}^N \log p(Y_i|X_i,W)
\end{aligned}
$$

### Final Layer

This network is the simplest to implement: add a distribution as the final layer. Typically we assume a Gaussian distribution because it is the easiest to understand and the training procedure is also quite simple.

$$\mathcal{N}(\mu(x), \sigma)$$

where $\mu$ is our point estimate and $\sigma$ is a **constant** noise parameter. Notice how the only parameter in our distribution is the mean function $\mu$. We assume that the $\sigma$ parameter is constant across every $\mu(x_i)$. This is known as a homoscedastic model. In this [blog post]() they classified this as a **known knowns**. When training, we can use the negative log-likelihood or the mean absolute squared loss function. In terms of applicability, this model won't be so difficult to train but it isn't the most robust model because in the real world, we won't find such a simple noise assumption. But it's a cheap way to add some uncertainty estimates to your predictions.

**Regression**

$$p($$

**Summary**

* Extends deterministic DNN to produce an output distribution, $p(y|x)$
* Uses maximum likelihood estimation to obtain the best model
* It is still time consuming training and can still overfit
* It can only quantify data uncertainty (aleatoric) 

### Heteroscedastic Noise Model

This network is very similar to the above model except we assume that the noise varies as a function of the inputs. 

$$\mathcal{N}(\mu(x), \sigma(x))$$

This accounts for the aleatoric uncertainty. So it's the same addition to the output layer mentioned above.is known as heterscedastic model. 



Again, in this [blog post]() they classified this as a **Known Unknowns**.

### Gaussian Process (Deep Kernel Learning)

## Bayesian Benchmarks

So there are a few Benchmark datasets we can look at to determine

Current top:

* MC Dropout
* Mean-Field Variational Inference
* Deep Ensembles
* Ensemble MC Dropout

Benchmark Repos:

* [OATML](https://github.com/OATML/bdl-benchmarks)
* [Hugh Salimbeni](https://github.com/hughsalimbeni/bayesian_benchmarks)

### Resources

* Neural Network Diagrams - [stack](https://softwarerecs.stackexchange.com/questions/47841/drawing-neural-networks#targetText=Drawing%20neural%20networks&targetText=Similar%20to%20the%20figures%20in,multilayer%20perceptron%20(neural%20network).)
* MLSS 2019, Moscow - Yarin Gal - [Prezi I](http://bdl101.ml/MLSS_2019_BDL_1.pdf) | [Prezi II](http://bdl101.ml/MLSS_2019_BDL_2.pdf)
* Fast and Scalable Estimation of Uncertainty using Bayesian Deep Learning - [Blog](https://medium.com/lean-in-women-in-tech-india/fast-and-scalable-estimation-of-uncertainty-using-bayesian-deep-learning-e312571042bb)
* Making Your Neural Network Say "I Don't Know" - Bayesian NNs using Pyro and PyTorch - [Blog](https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd)
* How Bayesian Methods Embody Occam's razor - [blog](https://medium.com/neuralspace/how-bayesian-methods-embody-occams-razor-43f3d0253137)
* DropOUt as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning - [blog](https://medium.com/@ahmdtaha/dropout-as-a-bayesian-approximation-representing-model-uncertainty-in-deep-learning-7a2e49e64a15)
* Uncertainty Estimation in Supervised Learning - [Video](https://www.youtube.com/watch?v=P4WUl7TDdLo&list=PLe5rNUydzV9QHe8VDStpU0o8Yp63OecdW&index=29&t=0s) | [Slides](https://github.com/bayesgroup/deepbayes-2019/tree/master/lectures/day6)


**Blogs**

* [Regression with Probabilistic Layers in TensorFlow Probability](https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf)
* [Variational Inference for Bayesian Neural Networks](http://krasserm.github.io/2019/03/14/bayesian-neural-networks/) (2019) | TensorFlow
* Brenden Hasz
	* [Bayesian Regressions with MCMC or Variational Bayes using TensorFlow Probability](https://brendanhasz.github.io/2018/12/03/tfp-regression.html)
	* [Bayesian Gaussian Mixture Modeling with Stochastic Variational Inference](https://brendanhasz.github.io/2019/06/12/tfp-gmm.html)
	* [Trip Duration Prediction using Bayesian Neural Networks and TensorFlow 2.0](https://brendanhasz.github.io/2019/07/23/bayesian-density-net.html)
* Yarin Gal
	* [What My Deep Model Doesn't Know...](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html)
* High Level Series of Posts
	* [Probabilistic Deep Learning: Bayes by Backprop](https://medium.com/neuralspace/probabilistic-deep-learning-bayes-by-backprop-c4a3de0d9743)
	* [When machine learning meets complexity: why Bayesian deep learning is unavoidable](https://medium.com/neuralspace/when-machine-learning-meets-complexity-why-bayesian-deep-learning-is-unavoidable-55c97aa2a9cc)
	* [Bayesian Convolutional Neural Networks with Bayes by Backprop](https://medium.com/neuralspace/bayesian-convolutional-neural-networks-with-bayes-by-backprop-c84dcaaf086e)
	* [Reflections on Bayesian Inference in Probabilistic Deep Learning](https://medium.com/@laumannfelix/reflections-on-bayesian-inference-in-probabilistic-deep-learning-416376e42dc0)

**Software**

* [TensorFlow Probability]()
	* [Edward2]()
* [PyTorch]()
	* [Pyro]()

**Papers**

* DropOut as Bayesian Approximation - [Paper](https://arxiv.org/pdf/1506.02142.pdf) | [Code]() | [Tutorial](https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/)
* Uncertainty Decomposition in BNNs with Latent Variables - [arxiv](https://arxiv.org/abs/1706.08495)
* Practical Deep Learning with Bayesian Principles - [arxiv](https://arxiv.org/abs/1906.02506)
* Pathologies of Factorised Gaussian and MC Dropout Posteriors in Bayesian Neural Networks - Foong et. al. (2019) - [Paper]()
* Probabilistic Numerics and Uncertainty in Computations - [Paper](https://arxiv.org/pdf/1506.01326.pdf)
* Bayesian Inference of Log Determinants - [Paper](https://arxiv.org/pdf/1704.01445.pdf)

**Code**

* [A Regression Master Class with Aboleth](https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html) 
* BNN Implementations - [Github](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
* A Comprehensive Guide to Bayesian CNN with Variational Inference - Shridhar et al. (2019) - [Github](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)