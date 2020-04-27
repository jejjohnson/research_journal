---
title: Overview
description: Overview of RBF Parameter experiment
authors:
    - J. Emmanuel Johnson
path: docs/projects/similarity/kernel_alignment_params
source: README.md
---
# Kernel Parameter Estimation


## Motivation

* What is Similarity?
* Why HSIC?
* The differences between HSIC
* The Problems with high-dimensional data

---

## Research Questions


!!! note "Demo Notebook"
    See [this notebook](notebooks/1.0_motivation.md) for a full break down of each research question and why it's important and possibly difficult.

---

### 1. Which Scorer should we use?

We are looking at different "HSIC scorers" because they all vary in terms of whether they center the kernel matrix or if they normalize the score via the norm of the individual kernels.

!!! info "Different Scorers"
    === "HSIC"

        $$\text{HSIC} = \frac{1}{n(n-1)}\langle K_xH,K_yH \rangle_F$$

        **Notice**: we have the centered kernels, $K_xH$ and no normalization.

    === "Kernel Alignment"

        $$\text{KA} = \frac{\langle K_x,K_y \rangle_F}{||K_x||_F||K_y||_F}$$

        **Notice**: We have the uncentered kernels and a normalization factor.

    === "Centered Kernel Alignment"

        $$\text{cKA} = \frac{\langle K_xH,K_yH \rangle_F}{||K_xH||_F||K_yH||_F}$$

        **Notice**: We have the centered kernels and a normalization factor.

---

### 2. Which Estimator should we use?

!!! info "Example Estimators"
    === "Scott"
        **Notice**: This method doesn't take into account the dimensionality

    === "Median"
        **Notice**: We can scale this by a factor 

---

### 3. Should we use different length scales or the same?

---

### 4. Should we standardize our data?

---

### 5. Summary of Parameters

<center>

|                     | Options                      |
| ------------------- | ---------------------------- |
| Standardize         | Yes / No                     |
| Parameter Estimator | Mean, Median, Silverman, etc |
| Center Kernel       | Yes / No                     |
| Normalized Score    | Yes / No                     |

</center>

---



## Experiments



### Parameter Grid - 1D Data

[**Notebook**](notebooks/2.0_preliminary_exp.md)

---

### Parameter Grid - nD Data

!!! todo
    Notebook doing some demos

---

### Mutual Information vs HSIC scores

!!! todo
    Notebook doing some demos

---

## Results

---

### Take-Home Message I

The median distance seems to be fairly robust in settings with different samples and dimensions. Scott and Silverman should probably avoided if you are not going to estimate the the parameter per feature.

---

### Take-Home Message II


It appears that the centered kernel alignment (CKA) method is the most consistent when we compare the score versus the mutual information of known distributions. HSIC has some consistency but not entirely. The KA algorithm has no consistency whatsoever; avoid using this method for unsupervised problems.