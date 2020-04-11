# Next Steps

So after all of this literature, what is the next step for the community? I have a few suggestions based on what I've seen:

#### 1. Apply these algorithms to different problems (other than dynamical systems)

It's clear to me that there are a LOT of different algorithms. But in almost every study above, I don't see many applications outside of dynamical systems. I would love to see other people outside (or within) community use these algorithms on different problems. Like Neil Lawrence said in a recent MLSS talk; "we need to stop jacking around with GPs and actually **apply them**" (paraphrased). There are many little goodies to be had from all of these methods; like the linearized GP predictive variance estimate for better variance estimates is something you get almost for free. So why not use it? 

#### 2. Improve the Kernel Expectation Calculations

So how we calculate kernel expectations is costly. A typical sparse GP has a cost of $O(NM^2)$. But when we do the calculation of kernel expectations, that order goes back up to $O (DNM^2)$ . It's not bad considering but it is still now an order of magnitude larger for high dimensional datasets. This is going backwards in terms of efficiency. Also, many implementations attempt to do this in parallel for speed but then the cost of memory becomes prohibitive (especially on GPUs). There are some other good approximation schemes we might be able to use such as advanced Bayesian Quadrature techniques and the many moment transformation techniques that are present in the Kalman Filter literature. I'm sure there are tricks of the trade to be had there.

#### 3. Think about the problem differently

An interesting way to approach the method is to perhaps use the idea of covariates. Instead of the noise being additive, perhaps it's another combination where we have to model it separately. That's what Salimbeni did for his latest Deep GP and it's a very interesting way to look at it. It works well too!


#### 4. Think about pragmatic solutions

Some of these algorithms are super complicated. It makes it less desireable to actually try them because it's so easy to get lost in the mathematics of it all. I like pragmatic solutions. For example, using Drop-Out, Ensembles and Noise Constrastive Priors are easy and pragmatic ways of adding reliable uncertainty estimates in Bayesian Neural Networks. I would like some more pragmatic solutions for some of these methods that have been listed above. **Another Shameless Plug**: the method I used is very easy to get better predictive variances almost for free.

#### 5. Figure Out how to extend it to Deep GPs

So the original Deep GP is just a stack of BGPLVMs and more recent GPs have regressed back to stacking SVGPs. I would like to know if there is a way to improve the BGPLVM in such a way that we can stack them again and then constrain the solutions with our known prior distributions. 

