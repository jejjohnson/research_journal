# [Identity Trick](https://www.shakirm.com/slides/MLSS2018-Madrid-ProbThinking.pdf)

I think the slides from the MLSS 2018 meeting is the only place that I have encountered anyone actually explicitly mentioning this Identity trick.

Given an integral problem: 

$$p(x) = \int p(x|z)p(z)dz$$ 

I can multiply by an arbitrary distribution which is equivalent to 1.

$$p(x)=\int p(x|z) p(z) \frac{q(z)}{q(z)}dz$$

Then I can regroup and reweight the integral

$$p(x) = \int p(x|z)\frac{p(z)}{q(z)}q(z)dz$$

This results in a different expectation that we initially had

$$p(x) = \underset{q(z)}{\mathbb{E}}\left[ p(x|z)\frac{p(z)}{q(z)} \right]$$

Examples:

* Importance Sampling
* Manipulate Stochastic gradients
* Derive Probability bounds
* RL for policy corrections

