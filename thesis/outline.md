# Thesis Outline


---
---
## Chapter I - Introduction



---
## 1.1 Earth Science in the Wild

### 1.1.1 Problems

* Earth Observation
* Climate 
* Extreme Events - Drought
* Ocean

### 1.1.2 Data 

* Remote Sensing
* Physical Models
* Generalized Models
* Emulation


### 1.1.3 Drawbacks

* What/Which? - best model?
* How? - how does the model work?
* Why? - get knowledge


---
## 1.2 - Fundamental ML Problems 

### 1.2.1 - Representations

* Discriminative
* Generative
* Information

### 1.2.2 - Interpretation

* Affect (Sensitivity)
* Noise 
* Uncertainty

### 1.2.3 - Understanding

* Correlation
* Dependence
* Causation

---
## 1.3 - This Thesis

### 1.3.1 - Investigation

* **Research Question:** Can we push forward the notion of uncertainty in EO applications?
  * Discriminative Approach - GPs
    * Sensitivity
    * Uncertainty
  * Generative Approach - RBIG
    * IT measures
* **Intermmediate Steps:** Investigations that needed to happen
  * Uncertain GPs 
  * HSIC Parameter Estimation
  * 
* **Applications:**
  * Earth Science Data Cubes
  * Ocean Data
  * Drought Indices
  * Climate Models


### 1.3.2 - OutReach

* Reproducibility
* Blog Posts
* PyCon (?)
  
### 1.3.3 - My Contributions

* **Code**
  > I try to include links to model zoos whenever possible with tutorials of how to do certain things from scratch.
* **Toy Examples**
  > I will do many toy examples to highlight important concepts.
* **Blogs**
  > This highlights things that I think the ML community should know
* **Sleeper Theorems**
  > There are many things that you should know to read the thesis. Just like many papers. And sometimes it's not possible to put them in the appendix. So I will put it in boxes.
* **Supplementary Material** 
  > I am a strong advocate of telling the story without the need to go through unnecessary mathematics. So I dump all necessary derivations in the appendix. Including Notation and supplementary material

### 1.3.4 - Organization

* **Part I** - Data Representation Approach
* **Part II** - Discriminative Modeling Approach
* **Part III** - Generative Modeling Approach

---
---
## Chapter II - Data Representation

---

### 2.1 Kernel Methods

* Theory
* Regression (Classification), (KRR, GPR, SVM)
* Dimensionality Reduction (KPCA, KECA, DRR)
* Density (KDE, Distribution Regression)
* Information Theory (Renyi Stuff)
* Similarity Measures - Covariance v.s. Correlation (HSIC, KA)

**Sleeper Theorems**

* Mercer's Theorem
* HSIC $\equiv$ MMD


---
### 2.2 Density Estimation

* Parametric
  * Gaussian
  * Mixture of Gaussians
* Classical
  * Binning (Histogram) - 
  * Kernel - Smooth
  * kNN - Adaptive
* Neural Density Estimation
  * Normalizing Flows
  * Density Destructor
  * Gaussianization
* Conditional Density Estimation

---
### 2.3 Neural Networks

* Discriminative Neural Networks
* Probabilistic Neural Networks
* Fully Bayesian Neural Networks

---
### 2.4 Other

* Generalized Linear Models
* Ensemble Methods

---
### 2.4 All Connected

* Kernel Entropy Components
* Conditional


---
### 2.3 Modeling Approaches

#### 2.3.1 Discriminative Models

#### 2.3.2 Generative Models


#### 2.2.3 Information Theory

* Signal
* ...
* Approximate 
* Measures (I, H, MI, TC)

##### Change of Variables




---
---
## Chapter III - Discriminative Model


---
### 1.1 Regression

#### 1.1.1 Kernel Ridge Regression

#### 1.1.2 Gaussian Process Regression

---
### 1.3 Sensitivity

#### 1.3.1 Concept

#### 1.3.2 Derivative

---
### 1.2 Uncertainty

---
### 1.4 Applications

* ESDC
  * Sampling
  * Principal Curves
  * ESDC - HSIC Sensitivity
* 

---
---
## Chapter IV - Generative Model


---
### 4.1 Outline

---
### 4.2 Probability

* Concepts
* Bayesian Formulation
* Change of Variables

**Sleeper Theorems**

* Variational Inference
* Jensen's Inequality
* 
* Change of Variables 

---
### 4.3 Generative Models

* Density Destructors
* Normalizing Flows

**Sleeper Theorems**

* MSE vs LL vs KLD

---
### 4.4 Information Theory


#### 4.4.1 IT Measures

* Signal
* ...
* Approximate 
* Measures (I, H, MI, TC)

#### 4.4.2 Estimators

* Gaussian
* KNN/KDP
* RBIG
* HSIC/KA


---
### 4.5 - Applications

#### 4.5.1 Spatial-Temporal Analysis

* EGU18 (**C**)
* CI19 (**W**)
* RBIG 4 RS (**J**)

#### 4.5.1 Droughts

* CI19 (**W**)
* RBIG 4 RS (**J**)

#### 4.5.3 Climate Models

* AGU (**C**)
* RBIG 4 RS (**J**)

---
### 4.6 - Lab Notebooks

* Change of Variables Proof of Concept
* RBIG - Step-by-Step
* 

---
### Kernel Derivatives (Regression)

* Regression: KRR, GPR, 
* Classification: SVM
* Feature Selection: O/KECA
* Dependence: HSIC, rHSIC
* Kernel Model Zoo


**Application**

* Regression: Sampling
* Regression: Sensitivity Analysis
* Input Uncertainty

**Appendix**

* Kernel Methods
* Theorem
* Sensitivity

---
### Input Uncertainty

* Extended Literature Review
* Connection with Kalman Filters
* GP Model Zoo (GPy, TF, GPyTorch)

**Context**

* Uncertainty in the Literature

**Application**

* IASI
* Ocean bbP
* Alvaro Data

**Appendix**

* Taylor Series
* Moment Matching
* Variational Inference

---
### IT Measures


**Application**

* Earth Science Data Cubes (spatial-temporal)
* Drought Indices
* Climate Models

---
### Dependence

* Covariance vs Correlation
* HSIC and Kernel Alignment
* Mutual Information 


---


---
### Appendices

**Mathematical Preliminaries**

* Linear Algebra
* Probability Theory

### 