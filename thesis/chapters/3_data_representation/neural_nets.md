# Neural Networks

So in terms of neural networks and uncertainty, I would say there are about 3 classes of neural networks ranging from no uncertainty to full uncertainty. They include:
* generic neural networks (NNs) which have no uncertainty
* Probabilistic Neural Networks (PNNs) which have uncertainty in the predictions
* Bayesian Neural Networks (BNNs) which have uncertainty on the weights as well

**Resources**
* Jee - [Prob CV](https://drive.google.com/file/d/1JMoWejUkOymaDK3G_yX0VAhVtesjqVCW/view?usp=sharing)
* Masterclass in regression - Aboleth - [docs](https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html)

---
## Weight Space

This neural network is vanilla with no uncertainty at all. We have no probability assumptions about our data. We merely want some function $f$ that maps our data from $X \rightarrow Y$. 


**Summary**

* Very expressive architectures
* Training yields the best model parameters
* Cannot quantify the output uncertainty or confidence

---

### Bayesian Treatment

---

## Probabilistic Neural Network

This class of neural networks are very cheap to produce. They basically attach a probability distribution on the final layer of the network. They don't have any probability distributions on and of the weights of the network. Another way to think of it is as a feature extractor that maps all of the data to a . Another big difference is the training procedure. Typically one would have to use a log-likelihood estimation but this isn't always necessary for PNN with simpler assumptions.

#### Learning: Maximum Likelihood

Given some training data $D=(X_i, Y_i), i=1,2,\ldots,N$, we want to learn the parameters $W$ by maximizing the log likelihood i.e.

$$
\begin{aligned}
W^*&= \underset{W}{\text{argmax}} \log p(D|W) \\
\log p(D|W) &= \sum_{i=1}^N \log p(Y_i|X_i,W)
\end{aligned}
$$

#### Final Layer

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

---

## Bayesian Neural Network



