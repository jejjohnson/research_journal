---
title: Overview
description: Overview of my similarity appendices
authors:
    - J. Emmanuel Johnson
path: docs/appendices/similarity
source: README.md
---
# Similarity?


## What is similarity?

> In words: 



## Overview of Methods


| Family          | Example Methods      |
| --------------- | -------------------- |
| Linear          | $\rho V$-Coefficient |
| Distance-Based  | Energy Distance, MMD |
| Shannon Entropy | D$_\text{KL}$        |

* All Linear methods are just regression - [notebook](https://eigenfoo.xyz/tests-as-linear/#8-Sources-and-further-equivalences)



| Method      | Property            |
| ----------- | ------------------- |
| Linear      | Uniform             |
| Distance    | Adaptive            |
| Kernels     | Adaptive, Smoothing |
| Information | Distribution        |


## Contents

### Linear Methods

In this document, I will go over the main linear ways that we can estimate similarity. They include the classic measures such as covariance and Pearson's correlation coefficient $\rho$. I will also go over how this can be extended to multivariate datasets with measures such as the congruence coefficient and the $\rho$V-Coefficient; multi-dimensional extensions to the original $\rho$.

* Theory: [Document](rv.md)
* Lab: [Colab Notebook](https://colab.research.google.com/drive/19bJd_KNTSThZcxP1vnQOVjTLTOFLS9VG)

---

### Distance-Based

This is a non-linear extension to the linear method which attempts to estimate similarity by distance formulas. This gives us a lot of flexibility with the distance metric that we choose (as long as they satisfy some key conditions). In the document, I will go over some of the common ones and how we go from linear to distance-metrics.

* Theory: [Document](distance.md)

---

### Kernel Methods

The kernel methods acts as a non-linear extension to the linear method and it generalizes the distance-based methods. These utilize kernel functions which act as non-linear functions which operate on the linear method and/or the distance metric. This adds an even higher level of flexibility. This is good or bad depending upon your perspective. There are two main candidates: the HSIC/Kernel Alignment and the Maximum Mean Discrepency (MMD) methods. They are related but the way the methods are calculated vary as will be explained above.

* Theory (HSIC): [Document](hsic.md)
* Theory (MMD): [Document](mmd.md)

---

### Variation of Information

The final method that we investigate is a different approach than the methods listed about: estimating the PDF of the dataset and then calculating Information theoretic measures.

* [Variation of Information](vi.md)

---

## Visualization

This is something that I didn't believe was so important until I had to explain my results. It was very clear it my head but I ended up showing a lot of different graphs and it was difficult for everyone to grasp the essence. Taylor Diagrams were originally used for univariate comparisons to show the Pearson correlation coefficient, the norm of the vector and the RMSE all in one plot. I show that this can be extended to the multivariate measures such as the $\rho$V-Coefficient, the Kernel Alignment Method and the Variation of Information methods.

* [Taylor Diagram](taylor.md)

---

## Resources


* [Comprehensive Survey on Distance/Similarity Measures between Probability Density Functions]() - Sung-Huyk Cha - International Journal of Mathematical Models and Methods in Applied Sciences, (...)
* [Multivariate Statistics Summary and Comparison of Techniques](http://www.umass.edu/landeco/teaching/multivariate/schedule/summary.handouts.pdf) - Presenation