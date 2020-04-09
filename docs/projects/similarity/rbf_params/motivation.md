# Motivation


In this document, I will be looking at the motivation behind this study and why we would like to pursue this further.

---

- [1D Dataset](#1d-dataset)
    - [Case I - Use the Same Length Scale](#case-i---use-the-same-length-scale)
    - [Case II - Use different length scales](#case-ii---use-different-length-scales)
    - [Verdict](#verdict)
- [2D Dataset](#2d-dataset)
    - [Case I - Use the Same Length Scale](#case-i---use-the-same-length-scale-1)
    - [Case II - Use different length scales](#case-ii---use-different-length-scales-1)
    - [Case II - Use different length scales](#case-ii---use-different-length-scales-2)
    - [Verdict](#verdict-1)
- [What Now?](#what-now)


## 1D Dataset

<center>
<img src="projects/similarity/rbf_params/pics/demo_sine.png" width="350">

**Fig I**: An example 1D distribution.
</center>

Let's take a simple 1D distribution: a sine curve. It is clear that there is a nonlinear relationship between them that cannot be captured (well) by linear methods. We are interested in looking at the dependence between $X$ and $Y$. We have the HSIC family of methods: HSIC, kernel alignment and centered kernel alignment. They are all very similar but there are some subtle differences. We will highlight them as we go through the overview. Let's take a generic approach and use the default HSIC, KA and CKA methods to try and estimate the dependence between $X,Y$. If we run the algorithm, we get the following results.

<center>

| **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| -------- | -------------------- | ----------------------------- |
| 0.00094  | 0.68847              | 0.58843                       |

</center>

Notice how all of the values are slightly difference. This is because of the composition of the methods. We can highlight the differences with a simple table.

<center>

| **Method**                | **Centered Kernel** | **Normalized** |
| ------------------------- | ------------------- | -------------- |
| HSIC                      | Yes                 | No             |
| Kernel Alignment          | No                  | Yes            |
| Centered Kernel Alignment | Yes                 | No             |

</center>

So each method has a slightly different formulation but they are mostly the same. So now the next question is: how do we estimate the parameters of the kernel used? Well the default is simply $\sigma=1.0$ but we know that this won't do as the kernel depends on the parameters of the kernel. In this case we are using the most commonly used kernel: the Radial Basis Function (RBF). Since this is a 1D example, I will use some generic estimators called the "Silverman Rule" and "Scott Rule". These are very commonly found in packages like `scipy.stats.gaussian_kde` or `statsmodels.nonparametric.bandwidth`. They are mostly used for the Kernel Density Estimation (KDE) where we need a decent parameter to approximate the kernel to get a decent density estimate. 

So what happens with the methods and the results?

<center>

|     | Silverman | Scott |
| --- | --------- | ----- |
| X   | 0.064     | 0.076 |
| Y   | 0.161     | 0.190 |

</center>

Notice that we get two different estimates for each dataset and for each estimator. They are relatively similar but they are not quite the same. So what do we do?

#### Case I - Use the Same Length Scale

<center>

| **Estimator** | $\sigma_{xy}$ | **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| ------------- | ------------- | -------- | -------------------- | ----------------------------- |
| Generic       | $1.0$         | 0.00094  | 0.68847              | 0.58843                       |
| Silverman     | $0.132$       | 0.132    | 0.612                | 0.494                         |
| Scott         | $0.113$       | 0.045    | 0.594                | 0.478                         |

</center>

Now we see that the values you get are quite different for all methods. What happens if we use different sigmas?

#### Case II - Use different length scales

| **Estimator** | $\sigma_{x}$ | $\sigma_{y}$ | **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| ------------- | ------------ | ------------ | -------- | -------------------- | ----------------------------- |
| Generic       | $1.0$        | $1.0$        | 0.00094  | 0.68847              | 0.58843                       |
| Silverman     | $0.132$      | $0.132$      | 0.132    | 0.612                | 0.494                         |
| Scott         | $0.113$      | $0.113$      | 0.045    | 0.594                | 0.478                         |
| Silverman     | $0.064$      | $0.161$      | 0.052    | 0.668                | 0.577                         |
| Scott         | $0.076$      | $0.190$      | 0.060    | 0.697                | 0.597                         |

Again, we have yet another set of sigma values and we don't really know which method to trust so how do we choose? 


#### Verdict

Well, hard to say as it depends on the parameters. Every researcher I've met who dealt with kernel methods seems to have a suggestion that they swear by but I never know who to follow. My thoughts is that we should use dedicated sigma values per dataset however, that still leaves us with other methods that we may want to try. So we're going to repeat the same experiment but with a 2D dataset and we will see that the difficult will increase again.

## 2D Dataset

<center>
<img src="projects/similarity/rbf_params/pics/demo_tstudent.png" width="350">

**Fig I**: An example 2D T-Student distribution.
</center>

For this experiment, we're going to take two 2D datasets each generated from a T-Student distribution. We will apply the same sequence as we did above and we will end the section by adding another option for picking the parameters.

#### Case I - Use the Same Length Scale

Firstly, assuming the same length scale for each.

<center>

| **Estimator** | $\sigma_{xy}$ | **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| ------------- | ------------- | -------- | -------------------- | ----------------------------- |
| Silverman     | $0.196$       | 0.0616   | 0.681                | 0.533                         |
| Scott         | $0.230$       | 0.0648   | 0.707                | 0.554                         |

</center>

There are some subtle differences but I would argue that these differences are less than the previous example. However, keep in mind that this is only two dimensions.

#### Case II - Use different length scales

Secondly, let's look at using different length scales.
<center>

| **Estimator** | $\sigma_{x}$ | $\sigma_{y}$ | **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| ------------- | ------------ | ------------ | -------- | -------------------- | ----------------------------- |
| Silverman     | $0.196$      | $0.196$      | 0.001    | 0.100                | 0.132                         |
| Scott         | $0.230$      | $0.230$      | 0.001    | 0.076                | 0.121                         |
| Silverman     | $0.194$      | $0.196$      | 0.001    | 0.101                | 0.132                         |
| Scott         | $0.229$      | $0.231$      | 0.001    | 0.076                | 0.121                         |

</center>
Again, we have yet another set of sigma values that are different between the method and the parameters. However, we have assumed that all data points can be estimated with the same length scale. But what about estimating each feature separately? It might actually make sense to try to estimate a length scale per feature.

#### Case II - Use different length scales

<center>

| **Estimator** | $\sigma_{x}$   | $\sigma_{y}$   | **HSIC** | **Kernel Alignment** | **Centered Kernel Alignment** |
| ------------- | -------------- | -------------- | -------- | -------------------- | ----------------------------- |
| Silverman     | $0.196$        | $0.196$        | 0.001    | 0.100                | 0.132                         |
| Scott         | $0.230$        | $0.230$        | 0.001    | 0.076                | 0.121                         |
| Silverman     | $0.194$        | $0.196$        | 0.001    | 0.101                | 0.132                         |
| Scott         | $0.229$        | $0.231$        | 0.001    | 0.076                | 0.121                         |
| Silverman     | $0.224, 0.224$ | $0.226, 0.226$ | 0.00095  | 0.07939              | 0.12150                       |
| Scott         | $0.264, 0.264$ | $0.265, 0.265$ | 0.00094  | 0.06002              | 0.11962                       |

</center>

So these values, I would argue, are roughly the same. But again, this is for 2 dimensions and also I have used fairly similar estimators. If I had used the mean or the median, this could have changed a lot.


#### Verdict

For the distributions, it seemed to be a little more consistent but with higher dimensions and more samples, these estimators start to fail. But then, we still don't have good alternative estimators.


## What Now?

I will be looking at the following:

<center>

|                     | Options                      |
| ------------------- | ---------------------------- |
| Standardize         | Yes / No                     |
| Parameter Estimator | Mean, Median, Silverman, etc |
| Center Kernel       | Yes / No                     |
| Normalized Score    | Yes / No                     |

</center>