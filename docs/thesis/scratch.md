# My Thesis: A Overview

These are some scratch notes for organizing my thesis.


---

## IDEAS to Add

* Two Views on Regression with PyMC3 and sklearn - [prezi](https://colcarroll.github.io/pydata_nyc2017/)

---

## Main Idea

1. Spatial-Temporal, High Dimensional, Complex Data - What do we do with it?
2. Data Inherent Features - Point Estimates, Spatial Meaning, Temporal Meaning
3. Understaing - Similarities &rarr; Correlations &rarr; Dependencies &rarr; Causation
4. ML Emulator Attributes - Sensitivity, Uncertainty, Scale, Error Propagation

---

## Part I - Sensitivity Analysis

> Opening up black-box models (i.e. kernel methods)


* We look at sensitivity methods in the context of kernel methods
* "Open the black box"
* Non-Bayesian Context
  * (KRR, SVM, KDE, HSIC)
  * Earth Data
* "Incomplete" Bayesian Context
  * Show examples in the context of a Bayesian Model (GPs + Emulation)

**Publication**

* SAKAME

#### Lab Notebooks

* GPs + GSA + Emulation -  $\phi$-Week

#### Tutorials

* Regression: KRR, OKRR, RFF
* Classification: SVM, RFF+SGD
* Density Estimation: KDE, OKECA
* Dependence Estimation: HSIC, rHSIC

---
---

## ML Problems

* Representations?
* Sensitivity
* Uncertainty Estimates
* Noise Characterization

---

## Background

* Representations (Kernels, Random Features, Deep Networks)
* Uncertainty
* Noise
* Analysis

---

## Key Concepts

* Representations - Kernels, Random Features, NNs|PNN|BNNs
* Similarity Measures - Naive, HSIC, IT
* Uncertainty - Epistemic, Aleatoric, Out-of-Distribution
* Methods - Discriminative (Model), Density Destructors (Density Estimation)

---

## Model-Based

* Representations
* Analysis - Derivative, Sensitivity
* Uncertainty Characterization - Output-Variance (eGP, eSGP), Input-Training (eVGP, BGPLVM)
* Applications
  * Emulation + Sensitivity
  * Multi-Output + Sensitivity


---

## Information-Based

* IT Measures
* Classic Methods - single-dimension/multivariate, mean, std dev, pt est. stats
* generative modeling - VAE, GAN, NF, DDD
* GAUSSIANIZATION
* Neural Networks, Deep GPs
* Noise Characterization
* Require Densities - Gaussian, Mixture, Histogram, KDE, Neural, RBIG
* Applications
  * climate model + noise + MI
  * sampling

---

## Applications

* Climate Models
  * Spatial Representations
  * Noise Characterization
  * Information Theory Estimates
* IASI
  * Error Propagation
  * Uncertainty Estimates
  * Multi-Output
  * Spatial-Temporal
* ARGP - BBP Data
  * Multi-Output
  * Spatial-Temporal
  * Sensitivity
  * Uncertainty
* Drought Variables
  * Temporal
* Emulation
  * Sensitivity
  * Uncertainty
