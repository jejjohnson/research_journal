---
title: Overview
description: All of my projects
authors:
    - J. Emmanuel Johnson
path: docs/projects
source: README.md
---
# Main Projects

---

## Similarity Measures

!!! abstract "Summary"
    I am very interested in the notion of similarity: what it means, how can we estimate similarity and how does it work in practice. Below are some of the main projects I have been working on which include an empirical study, some applications and some software that was developed.

??? info "Kernel Parameter Estimation"
    <!-- !!! abstract "Summary" -->
    In this project, I look at how we one can estimate the parameters of the RBF kernel for various variations of the HSIC method; kernel alignment and centered kernel alignment. Unsupervised kernel methods can suffer if the parameters are not estimated correctly. So I go through and empirically look at different ways we can represent our data and different ways we can estimate the parameters for the unsupervised kernel method.

    I investigate the following questions:

    * Will **standardizing** the data beforehand affect the results?
    * How does the **parameter estimator** affect the results?
    * Which **variation of HSIC** gives the best representation of the similarity (center the kernel, normalize the score)?
    * How does this all compare to **mutual information** for known high-dimensional, multivariate distributions?

    **Important Links**:

    * [Main Project Page](kernel_alignment_params/README.md)
    * [LaTeX Doc](/)
    * [FFT Talk](../talks/2020_fft_01_31_hsic_align.md)

??? info "Information Measures for Climate Model Comparisons"
    <!-- !!! abstract "Summary" -->
    In this project, I used a Gaussianization model to look compare some CMIP5 models the spatial-temporal repre

    **Important Links**:

    * [Main Projecct Page](/)
    * [LaTeX Doc](/)
    * [FFT Talk](../../talks/2019_agu_rbigclima.md)


??? info "Information Measures for Drought Factors"
    **Important Links**:

    * [Main Projecct Page](/)
    * [LaTeX Doc](/)
    * Paper: [Climate Informatics](/)
    * Poster: [Climate Informatics](/)
    * [Phi-Week Talk](../../talks/2019_phiweek_rbigad.md)


??? info "PySim"
    Some highlights include:

    * Scikit-Learn Format to allow for pipeline, cross-validation and scoring
    * The HSIC and all of it's variations including the randomized implementation
    * Some basics for visualizations using the Taylor Diagram
    * Some other methods for estimating similarity

    **Important Links**:

      * [Github Repository](https://github.com/jejjohnson/pysim.git)

---

## Uncertainty Quantification

!!! info "Projects"
    * [Input Uncertainty for Gaussian Processes](/)
    * [Gaussianization Models in Eath Science Applications](/)
    * [Gaussian Process Model Zoo](/)
    * [RBIG 1.1 Python Package](/)
    * [RBIG 2.0 Python Package](/)

---

## Kernel Methods and Derivatives

!!! info "Projects"
    * [Kernel Derivatives Applied to Earth Science Data Cubes](/)
    * [Derivatives for Sensitivity Analysis in Gaussian Processes applied to Emulation](/)

---

## Machine Learning for Ocean Applications

!!! info "Projects"
    * [ARGO Floats MultiOutput Models](/)
    * [Ocean Water Types Regression](/)
    * [Ocean Water Types Classification](/)
