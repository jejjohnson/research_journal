# Bayesian Neural Networks Resources


---
## Resources

* [Paper Summaries](https://github.com/fregu856/papers)
* [Recent Papers](https://github.com/mcgrady20150318/BayesianNeuralNetwork)


---
## Background


---
## Software



---
## Coding Practices

So I include this section to try and highlight some repos or blog posts that did a really good job of coding the algorithms listed. They also did it in a very modular and clear way.

* [Tutorial: Dropout as Regularization and Bayesian Approximation](https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/)
  > They use PyTorch and implement the MC dropout Layer and Bayesian CNN Layer.
* 

---
## Algorithms

## Bayesian

* http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
* http://krasserm.github.io/2018/03/19/gaussian-processes/


## Code

* BNN Implementations - [Github](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
* A Comprehensive Guide to Bayesian CNN with Variational Inference - Shridhar et al. (2019) - [Github](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)



---
## Bayesian Layers

I put it as Bayesian layers just to separate the individual components from the full models. For example, dropout is a layer that you can add to your neural network but it's not the full change in model architecture like a U-Net or a ResNet.


### Drop-Out

A very cheap way to constrain your models and it provides very good uncertainty estimates of your predictions. It doesn't even have to be a BNN.


* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://medium.com/@ahmdtaha/dropout-as-a-bayesian-approximation-representing-model-uncertainty-in-deep-learning-7a2e49e64a15)



### Noise Contrastive Priors

This is a very cheap way to account for uncertainty in your inputs. You do a concatenation of your data and your perturbed data as the input to the neural network.

* [Project Page](https://danijar.com/project/ncp/)

**Sample Code Block**

I will give a sample code block to demonstrate how simple this can be using the edward2 library (backend TensorFlow).

```python
batch_size, dataset_size = 128, 1000
features, labels = get_some_dataset()
inputs = keras.Input(shape=(25,))

"""This layer takes the noise model you recommend and then adds a concatenation
of that perturbed data as the input to the next model."""
x = ed.layers.NCPNormalPerturb()(inputs)  # double input batch

# Neural Network as per normal
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
means = ed.layers.DenseVariationalDropout(1, activation=None)(x)  # get mean
means = ed.layers.NCPNormalOutput(labels)(means)  # halve input batch
stddevs = tf.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means, stddevs])

```

---
## Algorithms

### Deep Ensembles

This method takes a number of different networks and then attempts to do some sort of averaging.

* [Paper](https://arxiv.org/abs/1612.01474) | [Blog Post](https://medium.com/@albertoarrigoni/paper-review-code-deep-ensembles-nips-2017-c5859070b8ce) | [Demo](https://nbviewer.jupyter.org/github/arrigonialberto86/deep-ensembles/blob/master/notebook/Playground.ipynb)

**Comments**:

* Where does a grad student get the computing/people power to try and train a large number of neural networks at the same time?


---
## Optimization Algorithms

* [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476) - Maddox et. al. (2019) | [Code](https://github.com/wjmaddox/swa_gaussian)
  > The authors implement an optimization algorithm called Stochastic Weighted Averaging-Gaussian(SWAG). They compare their implementation to standard algorithms such as MC-Dropout, KFAC-Laplace (which is a member of the natural gradient family) and temperature scaling.