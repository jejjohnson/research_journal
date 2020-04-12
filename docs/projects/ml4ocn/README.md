# Overview

I present some of my findings for working on machine learning applied to ocean applications.

## Documents

### Global Data

* Notebook I - Loading the Preprocessed Data - [ipynb](ARGO_project/global_data/1_load_processed_data.md)
* Notebook II - Loading the Preprocessed Data - [ipynb](ARGO_project/global_data/2_ml_algorithms.md)

---

## Walk-Throughs

These are the fully fledged notebooks which will go through all of the preprocessing steps that were done in order to achieve the results we achieved.

1. Explore the Data
2. Preprocessing - PCA Components Analysis, Column Transformations
3. End-to-End Training Example - RF, LR, MLP scikit-learn
4. Output Maps (Validation Floats)
5. Results
   1. Line Plots - Variance in Profiles
   2. Residual Plots
   3. Joint Distribution Plots

---

## Ideas

**BaseLine Models**: sklearn models
* Random Forests
* Linear Regression
* MLP

**SOTA Models**: Outside
* CatBoost
* XGBoost

**Bayesian Models**
* Keras:
  * Bayesian Layers (Edward)
  * Linear Regression (TF Probability)
  * Deep Kernel Learning (TF Probability, Bayesian Layers)

**GPyTorch Explore**
* Exact GP
* Sparse Variational GP
* Deep GP
* DKL - Sparse

### Tutorials

**Feature Extraction**
* Column Transformations (scikit-ify everything)
* PCA Trick

**CrossValidation**
* Model Only
* PreProcessing

**Model Training**
* Cross Validation (Random, GridSearch)
* Training Tips
  * [Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)
  * [Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py)

**Assessment**
* Interpretability
  * Permutation Maps
  * Sensitivity Analysis

**Visualization**
* Matplotlib - Complete Customization
* GeoPandas
* XArray
* HoloViews / GeoViews
* DataShader
