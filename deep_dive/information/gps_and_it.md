# Gaussians and Gaussian Processes

In this summary, I will be exploring what makes Gaussian distributions so special as well as the relation they have with IT measures. I'll also look at some extensions to Gaussian Processes (GPs) and what connection they have have with IT measures.

---
### GPs, Entropy, Residuals


> Easy to compute stuff out of a GP (which is a joint multivariate Gaussian with covariance K) would be:
> 1) Differential) entropy from the GP:
> $$H(X) = 0.5* log( (2*\pi*e)^n * \text{det}(K_x) )$$
> 
> where K_x is the kernel matrix (a covariance after all in feature space).
> https://en.wikipedia.org/wiki/Differential_entropy#Properties_of_differential_entropy
> http://www.gaussianprocess.org/gpml/chapters/RW.pdf  (see e.g. A.5, eq. A.20, etc)
> 
> 2) And then I remembered that the LS error estimate could be bounded, and since a GP is after all LS regression in feature space, maybe we could check if the formula is right:
> 
> $$MSE = E[(Y-\hat Y)^2] >= 1/(2*pi*e) * exp(H(Y|X))$$
> 
> https://en.wikipedia.org/wiki/Conditional_entropy
> that is, MSE obtained with the GP is lower bounded by that conditional entropy estimate from RBIG.
> 
> Some building blocks to start with connecting dots... :)
> 
> Jesus
> Intuition said that "the bigger the conditional entropy, the bigger the residual uncertainty is -> the bigger the MSE should be (no matter the model)"
> Cool wiki formula: MSE >= exp(conditional)



$$H(X_y|X_A)=\frac{1}{2} \log \left(\nu_{X_y|X_A}^2 \right) + \frac{1}{2} \left( \log(2\pi) + 1 \right)$$

where:

* Conditional Variance: $\nu^2_{y|A}=K_{yy}- \Sigma_{y|A}\Sigma_{AA}^{-1}\Sigma_{Ay}$

**Unread Stuff**:

* Conditional Entropy in the Context of GPs - [Stack](https://stats.stackexchange.com/questions/388761/conditional-entropy-in-the-context-of-gaussian-processes)
* Entripy of a GP (Log(det(Cov))) - [stack](https://stats.stackexchange.com/questions/377794/entropy-of-a-gaussian-process-logdeterminantcovariancematrix)
* Get full covariance matrix and find its entropy - [stack](https://stackoverflow.com/questions/53345624/gpflow-get-the-full-covariance-matrix-and-find-its-entropy)


### GPs and Causality

* Intro to Causal Inference with GPs - [blog I](https://mindcodec.ai/2018/10/22/an-introduction-to-causal-inference-with-gaussian-processes-part-i/) | [blog II](https://mindcodec.ai/2018/12/09/an-introduction-to-granger-causality-using-gaussian-processes-part-ii/)
