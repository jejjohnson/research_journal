# Fast Friday Talk

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

## An empirical investigation of Kernel Alignment parameters

* Date: 2020 - 01 - 31
* Day: Friday
* Time: 1200

---

### Abstract

When we have lots of data and want to make comparisons, we need methods to give us an overall summary. A very popular method is to measure the covariance and/or correlation which tackles this problem from a variability perspective. It's good but there are limitations to this method: 1) It can only handle linear relationships and 2) It's not clear how it applies to multivariate/multi-dimensional data. 

We can use a nonlinear kernel function on this covariance matrix which addresses some of the limitations that the covariance/correlation measures exhibit. However, there is a problem that all kernel methods have: the parameters of the kernel function. For a supervised learning problem such as regression or classification, you have an objective that you want to minimize so it's clear which parameters you should use. However in unsupervised settings, there is no objective so it's not clear how we pick the parameters for the kernel function.  

We do a detailed empirical analysis using different kernel parameter initialization schemes, for different toy datasets, and with different sample and dimension sizes. The obvious question we want to answer is: which kernel parameter should I use? And the answers that we found was as expected: it depends... But the nevertheless, the results are reassuring. 


---

### Slides

<p>
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQVUd4JoDa_5UzbsAjjkBRuxb4W8SrMQqYiMrXCWtTnyBe8sLN48MvxAG_zEy7fW_HoZyuzSGYzM3Ly/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</p>
