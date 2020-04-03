# Exponential Family of Distributions



This is the close-form expression for the Sharma-Mittal entropy calculation for expontial families. The Sharma-Mittal entropy is a generalization of the Shannon, Rényi and Tsallis entropy measurements. This estimates Y using the maximum likelihood estimation and then uses the analytical formula for the exponential family.



**Source Parameters, $\theta$**

$$\theta = (\mu, \Sigma)$$

where $\mu \in \mathbb{R}^{d}$ and $\Sigma > 0$

**Natural Parameters, $\eta$**

$$\eta = \left( \theta_2^{-1}\theta_1, \frac{1}{2}\theta_2^{-1} \right)$$

**Expectation Parameters**



**Log Normalizer, $F(\eta)$** 

Also known as the log partition function.

$$F(\eta) = \frac{1}{4} tr( \eta_1^\top \eta_2^{-1} \eta) - \frac{1}{2} \log|\eta_2| + \frac{d}{2}\log \pi$$


**Gradient Log Normalizer, $\nabla F(\eta)$**

$$\nabla F(\eta) = \left( \frac{1}{2} \eta_2^{-1}\eta_1, -\frac{1}{2} \eta_2^{-1}- \frac{1}{4}(\eta_2^{-1}-\eta_1)(\eta_2^{-1}-\eta_1)^\top \right)$$

**Log Normalizer, $F(\theta)$** 

Also known as the log partition function.

$$F(\theta) = \frac{1}{2} \theta_1^\top \theta_2^{-1} \theta + \frac{1}{2} \log|\theta_2| $$

**Final Entropy Calculation**

$$H = F(\eta) - \langle \eta, \nabla F(\eta) \rangle$$


## Resources

* A closed-form expression for the Sharma-Mittal entropy of exponential families - Nielsen & Nock (2012) - [Paper]()
* Statistical exponential families: A digest with flash cards - [Paper](https://arxiv.org/pdf/0911.4863.pdf)
* The Exponential Family: Getting Weird Expectations! - [Blog](https://zhiyzuo.github.io/Exponential-Family-Distributions/)
* Deep Exponential Family - [Code](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/deep_exponential_family.py)
* PyMEF: A Framework for Exponential Families in Python - [Code](https://github.com/pbrod/pymef) | [Paper](http://www-connex.lip6.fr/~schwander/articles/ssp2011.pdf)
