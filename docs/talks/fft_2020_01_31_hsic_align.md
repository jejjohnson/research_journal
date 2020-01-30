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

Kernel methods are class of machine learning algorithms solve complex nonlinear functions by a transformation facilitated by flexible, and expressive representations through kernel functions. For classes of problems such as regression and classification, there is an objective which allows provides a criteria for selecting the kernel parameters. However, in unsupervised settings such as dependence estimation where we compare two separate variables, there is no objective function to minimize or maximize except for the measurement itself. Alternatively one can choose the kernel parameters by a means of a heuristic but it is unclear which heuristic is appropriate for which application. 

The Hilbert-Schmidt Independence Criterion (HSIC) is one of the most widely used kernel methods for estimating dependence but it is not invariant to isotropic scaling for some kernels and it is difficult to interpret because there is no upper bound. The Kernel Alignment (KA) method is a normalized version of HSIC which alleviates the isotropic scaling issue as well as gives a more interpretable output value.  In this work we demonstrate how the kernel parameter for maximum value of the HSIC measures changes depending on the toy dataset and the amount of noise present. We also demonstrate how the methods compare when evaluated on known distributions where the analytical mutual information is available.


---

### Slides

<p>
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQVUd4JoDa_5UzbsAjjkBRuxb4W8SrMQqYiMrXCWtTnyBe8sLN48MvxAG_zEy7fW_HoZyuzSGYzM3Ly/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</p>
