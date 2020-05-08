---
title: Multi-Output
description: Notes for Multi-Output Learning
authors:
    - J. Emmanuel Johnson
path: docs/appendices/ml
source: multioutput.md
---
# Multi-Output / Multi-Task learning

So in this setting, we assume that we have a feature vector $\mathbf{x} \in \mathbb{R}^D$ and an output vector $\mathbf{y} \in \mathbb{R}^P$.


## Talk from Andrei Karpathy

### Architectures

---

#### One Model per Task

> We use one model per task. So we have $P$ independent $f(\cdot)$'s for each output $\mathbf{y}$. $F = \left[f_1, f_2, \ldots, f_p   \right]$

**Pros**

* Decoupled Functionality
* Easier to manage

**Cons**

* Expensive to calculate at test time
* No feature sharing
* Potential overfitting with data concernts

---

#### One Model for all

**Pros**

* Cheaper at test time
* Tasks help capacity

**Cons**

* Fully coupled functionality
  * expensive to fine-tine because you need to retrain the entire model every time
* Tasks fight for capacity
  * One task might be easier to train whereas another might not be. If there is no dedicated loss function that addresses this (correctly) then you don't discriminate between tasks very well.

---

## Loss Functions

When we minimize something, we need a loss function. If we have multiple tasks, then we need to ensure we have a loss function that best describes our needs. This is another *static* consideration.

$$
\mathcal{L}(\theta) = \min_\theta \alpha l_1 + \beta l_2 + \ldots + \zeta l_P
$$

where $l$ is a loss term for a single output/task $p$. **Note**: This method works well for 2 problems but for $p>2$ this doesn't work as well.

Some examples where we can derive loss functions

**Scales**

We can have a loss function that considers different scales, e.g. if we have a regression and a classification task. Then we need to scale each of the tasks appropriately. If I have a regression task and a 10-classification tasks, how do you scale this appropriately?

**Importance**

We may have specific goals or specific problems that we want to do better at than others. For this, we need to add some sort of weight in order to focus on that problem.

**Convergence**

Some tasks may be easier or harder. The easier tasks will probably converge faster whereas the harder tasks will not. So how do you propose early stopping criteria? It becomes quite difficult when you have multiple tasks which may require multiple stopping criteria.

**More/Less Data**

Some tasks will have more data which might increase the convergence time and/or hinder the training of other tasks. So you need to adjust your loss function accordingly.

**More Noise**

More noise means more regularization. But you need to know the amount of noise a priori and need to adjust accordingly.

---

## Training Dynamics

### Sampling

Do we do **within-task** sampling and **across-task** sampling? If we know that two tasks are very sensitive to certain features, it makes sense for them to use only those features.

**Amortized**?

**Team Management**

If you have a giant network and you want to fine-tune certain bits and pieces of your network, then it becomes difficult to allocate resources and allow people to only change certain parts. For example, if each person tries to change one little section, then this may lead to really complex networks that are no longer able to reproduce the best results. In a **Transfer Learning**, we definitely need to consider the final product.

---

## Literature

**Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation** - Liu et. al. (2019) - [arxiv](https://arxiv.org/abs/1901.02985) | [PyTorch](https://github.com/NoamRosenberg/AutoML)

> This paper searches the space of neural network for the best neural network using brute force search techniques.

**Which Tasks Should Be Learned Together in Multi-task Learning?** - Standley et. al. (2019) - [arxiv](https://arxiv.org/abs/1905.07553)