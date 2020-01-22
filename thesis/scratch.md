# My Thesis: A Overview

These are some scratch notes for organizing my thesis.

---

## Main Idea

1. Spatial-Temporal, High Dimensional, Complex Data - What do we do with it?
2. Data Inherent Features - Point Estimates, Spatial Meaning, Temporal Meaning
3. Understaing - Similarities &rarr; Correlations &rarr; Dependencies &rarr; Causation
4. ML Emulator Attributes - Sensitivity, Uncertainty, Scale, Error Propagation

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
