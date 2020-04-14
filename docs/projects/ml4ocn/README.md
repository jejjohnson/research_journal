---
title: Overview
description: Overview of my ML4OCN Research
authors:
    - J. Emmanuel Johnson
path: docs/projects/ML4OCN
source: README.md
---
# Overview

I present some of my findings for working on machine learning applied to ocean applications.

## ARGO Project

!!! summary "Summary"
    In this project, we are trying to predict profiles of $Bbp$. This is a multi-output ML problem with high dimensional inputs and high dimensional outputs.

**Note**: You can find the repository with all of the reproducible code, specific functions and notebooks at [github.com/IPL-UV/ml4ocean](https://github.com/IPL-UV/ml4ocean).

---

### Subset Dataset

We look at the North Atlantic and SubTropical Gyre regions to try and get an idea about how some ML methods might perform in this scenarios. As a first pass, we did minimal preprocessing with some standard PCA components reduction and we were ambitious and attempted to predict all 273 depths for the profiles. We had some success as we were able to do a decent job with Random forests and MultiLayer Perceptrons.

!!! fire "Relevant Materials"
    * ISPRS 2020 Publication - **Pending**

??? example "Todo"
    * 0.0 - Data Exploration Notebook
    * 1.0 - Data Preprocessing Notebook
    * 2.0 - Loading the Preprocessed Data
    * 3.0 - Baseline ML Algorithm
    * 4.0 - Visualization & Summary Statistics

---

### Global Dataset

After we found success in the subset data, we decided to go for the global dataset and see how we do. We did reduce the problem difficulty by predicting only 19 depth levels instead of $\sim$273.

!!! check "Notebooks"
    * 2.0 - [Loading the Preprocessed Data](ARGO_project/global_data/1_load_processed_data.md)
    * 3.0 - [Baseline ML Algorithm](ARGO_project/global_data/2_ml_algorithms.md)

??? example "Todo"
    * 0.0 - Data Exploration Notebook
    * 1.0 - Data Preprocessing Notebook
    * 4.0 - Visualization & Summary Statistics
