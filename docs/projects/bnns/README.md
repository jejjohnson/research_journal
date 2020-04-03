# Bayesian Neural Networks Working Group

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: [isp.uv.es/working_groups/bnn](https://isp.uv.es/working_groups/bnn)

---

### Summary

This is the working group page for exploring Bayesian Neural Networks (BNNs). Recently BNNs have started to become popular in the Machine learning literature as well as the applied sciences literature. Most research groups are interested because of the 'principled' approach to handling uncertainty within your data. Many traditional ML approaches don't account for uncertainty and make this The adoption of Deep learning methods and easy-to-use open-source software has also aided in the growning popularity. It is now easier to implement and try various models without having to do things from scratch. It's a good time to see if the Bayesian methodology works for your problem as the field has started to progress.

In this working group we will working with the Bayesian methodology from 3 perspectives: **Theory**, **Practice**, and **Application**. Even though there are 3 parts, we will be heavily driven by the application portion. After talking with the lab, we have a list of possible applications where we think BNNs would be appropriate. This could determine the direction of our exploration as a principled approach to dealing with your problem does require us to think about our data and which approaches will be appropriate.  We will also adopt a balance between the methods that seem to work in practice (e.g. Drop-Out, Ensembles) and the methods that "would be nice" (e.g. VI-based layers, Deep Gaussian Processes, SWAG). This means that we will include things that approximate NNs such as drop-out and architectures that are a mixture of standard NNs and probabilistic models. I outline each of the sections in detail below.

---

### Sections Outline

**Theory**

We will look at some of the staple papers which started the BNN approach as well as some of the SOTA approaches. In addition to Bayes related material, we will also take a look at some things related to uncertainty and neural networks.

**Practice**

We will go over some key probabilistic programming aspects. This is different than the standard Neural network architecture and can be a bit difficult to fully grasp. I think with adequate training in the software side of things, your life will be must easier and you will be able to **correctly** and **efficiently** implement algorithms and concepts.

**Applications**

This will be somewhat application driven, at least for the practical aspects. In the end, the groups have all come with hopes that they can use some of these techniques in the near future.
We currently have the following pending applications:
* Emulation Data
* Ocean Data (Multi-Output)
* Medical Data
* Gap Filling Data

---

### Format (**TBD**)

I would like to balance the 3 things I've mentioned above. I would like to spend time in the meetings discussing the theory and concepts. And then we can have a few sessions discussing some programming concepts to ensure that we can be doing practice on the way. Perhaps the individual groups can work on the applications in their free time.

---

### Requirements

This is not a beginners working group so there will be some expectations from the people attending if they're going to participate. They are not strictly required, but I suggest them because I think it would make the experience better for **everyone**.

* Familiarity with Bayesian Concepts (Prior, Likelihood, Posterior, Evidence, etc)
* Prior Programming experience; preferably in Python (**practical sessions**) 
* Familiarity with Neural networks and the terminology (gradients, loss, optimization, etc)

---

### Logistics

**When**
* Start: February, 2020
* Duration: TBD

**Where**
* ISP Open Office

**Leads**
* J. Emmanuel Johnson
* Kristoffer Wickstrom





---

## Resources

**Literature**
* [Papers](./theory/papers.md)
* [SOTA](./theory/sota.md)

**Resources**
* [Videos](./other/videos.md)

**Software**
* DL Frameworks
  * [Overview](./code/overview.md) 
  * [TensorFlow](./code/tensorflow.md)

