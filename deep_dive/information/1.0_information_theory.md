# Information Theory

In this report, I will be outlining what exactly Information theory is and what it means in the machine learning context. Many times when people ask me what I do, I say that I look at Information theory (IT) measures in the context of a generative network. But sometimes I have a difficult time convincing myself that these measures are actually useful. I think this is partially because I don't fully understand the magnitude of IT measures and what they can do. So this post is designed to help me (and others) really dig deep into the space of information measures and I hope this will help someone else who is also interested in understanding IT without any formal classes. This post will not be formula heavy and will instead focus on concepts. See the end of the post for some additional references where you can explore each of these ideas even further.

---
## Definition: What is Information?

Firstly, let's start with the definition of **information**. This is a difficult definition and I've seen many different explanations which are outlined below.

> The amount of surprise, i.e. I'm surprised when something that doesn't usually happen, happens. - *Jesus Malo, 2019*

This is my first definition which really resonates with me till this day. I think the concept of *surprise* is quite easy to grasp because it is intuitive. The first example I have is when I google the answer to my question. Things that are informative

>  Information is what remains after every iota of natural redundancy has been squeezed out of a message, and after every aimless syllable of noise has been removed. It is the unfettered essence that passes from computer to computer, from satellite to Earth, from eye to brain, and (over many generations of natural selection) from the natural world to the collective gene pool of every species. - James V Stone, 2015

This is a very good general definition of information from a broad perspective. It is very involved so it stresses how information can be thought of from many different perspectives including remote sensing, neuroscience and biology. It also tries to hint at the distinction between useful information and noise: signal (useful) and noise (useless). This definition comes from the first chapter of the book written by James Stone (2015). I highly recommend you read it as it gives the best introduction to information theory that I have ever seen.

> Information is the resolution of uncertainty. - Shannon C, 1948

This is the classic definition of uncertainty given by the creator himself. ...

> 

> In information theory, one variable provides information about another variable when knowledge of the first, on average, reduces uncertainty in the second. - *Cover & Thomas, 2006*

This is technically the definition of mutual information (MI) which is the extension of entropy to 2 or more variables. However, I think this definition actually captures almost all aspects of the information theory measures



### Concrete Definitions	

So we have expressed how these methods work in words but I am 

#### Deterministic Perspective

> The number of data points that one needs to represent a signal without an error. - *Gabor, 1946*

$$\text{Information} \propto \Delta x \cdot \Delta f$$

Units: Wavelets (Log ones)


#### Probabilistic Perspective

> The reduction in uncertainty that was caused by the knowledge of a signal. - *Shannon, 1948*

$$\text{Information} \propto \int_{\mathcal{X}} p(x) \cdot \log_2 \frac{1}{p(x)} \cdot dx$$

Units: Bits

In this definition, the uncertainty is related to the volume of the PDF (more volume means more uncertainty).

> Information theory makes statements about the shapes of probability distributions.
> 
---
### Interpretable Quantities: Bits



> **Key Idea**: One bit is the amount of information required to choose between two equally probable alternatives



> **Key Idea**: If you have $n$ bits of information then you can choose from $m=2^n$ equally probable alternatives. Equivalently, if you have to choose between $m$ equally probable alternatives, then you need $n=\log_2 m$ bits of information.



> **Key Idea**: A bit is the amount of information required to choose between two equally probable alternatives (e.g. left/right), whereas a binary digit is the value of a binary variable, which can adapt one of two possible values (i.e. 0/1).



---

## Motivation: Why use IT measures?

I often wonder why should one use information theory measures in the first place. What makes IT a useful tool in the different science fields. I found this introduction from [1] to have the best characterization of why one should use IT measures as they outline a few reasons why one can use IT measures.

1. > **Information theory is model independent**
  
   This statement is very powerful but they basically mean that we do not need to make any assumptions about the structure or the interactions between the data. We input the data and we get out a quantitative measure that tells us the relationship within our data. That allows us to be flexible as we should be able to capture different types of phenomena without the need to make assumptions and assumed models that could result in limiting results. No need to make a pre-defined model. It sounds nice as a first step analysis.
   
   **Warning:** *Often, we do need to make assumptions about our data, e.g. independence between observations, constant output, and also the bin size that is needed when capturing the probability distribution. These parameters can definitely have an impact on the final numbers that we obtain.*

2. > **Information theory is data independent**
  
   This statement means that we can apply IT measures on any type of data. We can also do certain transformations of said data and then apply the IT measures on the resulting transformation. This allows us to monitor how the information changes through transformations which can be useful in applications ranging from biological such as the brain to machine learning such as deep neural networks.

3. > **Information theory can detect linear and nonlinear interactions**
  
   This speaks for itself. As we get more and more data and get better at measuring with better sensors, we can no longer describe the word with linear approximations. The relationships between variables are often non-linear and we need models that are able to capture these nonlinear interactions. IT measures can capture both and this can be useful in practice.

4. > **Information theory is naturally multivariate**
  
   In principle, IT measures are well-calibrated enough to handle large multivariate (and multi-dimensional) systems. The easiest number of variables to understand include 2-3 variables but the techniques can be applied to a higher number of variables.

   **Warning:** *In practice, you may find that many techniques are not equipped to handle high-dimensional and high-multivariate simply because the method of PDF estimation becomes super difficult with big data. This is essential to calculate any kind of IT measure so this can end up being the bottleneck in many algorithms.*

5. > **Information theory produces results in general units of bits (or nats)**
  
   The units from IT measures are universal. No matter what the inputs are of the units, we will get the same amount of information in the output. This allows for easy comparisons between different variables under different transformations.

   **Warning:** *The units are universal and absolute but it does not mean they convey meaning or understanding. I think it's important to look at them from a relative perspective, e.g. IT measure 1 has 2 bits which is higher than IT measure 2 which has only 1 bit.*

Overall, we can see that IT measures can handle multi-dimension and multivariate data, are model-free and have universal units of comparison. They are dependent on accurate PDF estimates and this. They have a strong argument for preliminary analysis of variables and variable interactions as well as complex systems.

[1]: Timme & Lapish, **A Tutorial for Information Theory in Neuroscience**, *eNeuro*, 2018

---
## What can IT measures tell us?

The authors from [1] also pose a good question for us to answer about why one should use IT measures. I agree with the authors that it is important to know what types of questions we can answer given some model or measurement. In the context of IT measures, they highlight a few things we can gain:

1. IT measures can quantify uncertainty of one or more variables.
  
   > It is possible to quantify how much a variable expected to vary as well as the expected noise that we use in the system.
2. IT measures can be used to restrict the space of possible models.
  
   > I have personal experience with this as we used this to determine how many spatial-spectral-temporal features were necessary for inputs to a model. We also looked at the information content of different variables and could determine how well this did.


We also can highlight things we cannot do:

1. We cannot produce models that describe how a system functions.
  
   > A good example is Mutual information. It can tell you how many bits of information is shared between data but not anything about **how** the system of variables are related; just that the variables are related to an extent.
2. The output units are universal so they cannot be used to produce outputs in terms of the original input variables.
  
   > In other words, I cannot use the outputs of the IT measures as outputs to a model as they make little sense in the real world.


Again, the authors highlight (and I would like to highlight as well): IT measures can be a good way to help build your model, e.g. it can limit the amount of variables you would like to use based on expected uncertainty or mutual information content. It is a quantified measure in absolute relative units which can help the user make decisions in what variables to include or what transformations to use.



---
### Information (Revisited)

Now we will start to do a deep dive into IT measures and define them in a more concrete sense.

$$I(x)= \underset{Intuition}{\log \frac{1}{p(x)}} = \underset{Simplified}{- \log p(x)}$$

The intuitive definition is important because it really showcases how the heuristic works in the end. I'm actually not entirely sure if there is a mathematical way to formalate this without some sort of axiom that we mentioned before about **surprise** and **uncertainty**.

We use logs because...

**Example Pt I: Delta Function, Uniform Function, Binomial Curve, Gaussian Curve**

### Entropy

This is an upper bound on the amount of information you can convey without any loss ([source](https://blog.evjang.com/2019/07/likelihood-model-tips.html)). More entropy means more **randomness** or **uncertainty**

$$H(X)=\int_{\mathcal{X}}p(x)\cdot \log p(x) \cdot dx$$

We use logs so that wee get sums of entropies. It implies independence but the log also forces sums.

$$H(Y|X) = H(X,Y)-H(X)$$

$$H(Y|X) = \int_{\mathcal{X}, \mathcal{Y}}p(x,y) \log \frac{p(x,y)}{p(x)}dxdy$$


#### Examples

**Example Pt II: Delta Function, Uniform Function, Binomial Curve, Gaussian Curve**

#### Under Transformations

In my line of work, we work with generative models that utilize the change of variable formulation in order to estimate some distribution with 

$$H(Y) = H(X) + \mathbb{E}\left[ \log |\nabla f(X)|\right]$$

* Under rotation: Entropy is invariant
* Under scale: Entropy is ...???
* Computational Cost...?

---
### Relative Entropy (Kullback Leibler Divergence)

This is the measure of the distance between two distributions. I like the term *relative entropy* because it offers a different perspective in relation to information theory measures.

$$D_{KL}(p||q) = \int_{\mathcal{X}}p(x) \cdot \log \frac{p(x)}{q(x)} \cdot dx \geq 0$$

If you've studied machine learning then you are fully aware that it is not a distance as this measure is not symmetric i.e. $D_{KL}(p||q) \neq D_{KL}(q||p)$.

#### Under Transformations

The KLD is invariance under invertible affine transformations, e.g. $b = \mu + Ga, \nabla F = G$

---
### Mutual Information

This is the reduction of uncertainty of one random variable due to the knowledge of another (like the definition above). It is the amount of information one r.v. contains about another r.v..

---
### Total Correlation

This isn't really talked about outside of the ML community but I think this is a useful measure to have; especially when dealing with multi-dimensional and multi-variate datesets. 

---
### PDF Estimation

For almost all of these measures to work, we need to have a really good PDF estimation of our dataset, $\mathcal{X}$. This is a hard problem and should not be taken lightly. There is an entire field of methods that can be used, e.g. autoregressive models, generative networks, and Gaussianization. One of the simplest techniques (and often fairly effective) is just histogram transformation. 
I work specifically with Gaussianization methods and we have found that a simple histogram transformation works really well. It also led to some properties which allow one to estimate some IT measures in unison with PDF estimation. Another way of estimating PDFs would be to look at kernel methods (Parezen Windows). A collaborator works with this methodology and has found success in utitlize kernel methods and have also been able to provide good IT measures through these techniques.



## References





---
## Supplementary Material

### GPs and IT








---
## References

#### Gaussian Processes and Information Theory


#### Information Theory


* Information Theory Tutorial: The Manifold Things Information Measures - [YouTube](https://www.youtube.com/watch?v=34mONTTxoTE)
* [On Measures of Entropy and Information](http://threeplusone.com/on_information.pdf) - 
* [Understanding Interdependency Through Complex Information Sharing](https://pdfs.semanticscholar.org/de0b/e2001efc6590bf28f895bc4c42231c6101da.pdf) - Rosas et. al. (2016)
* The Information Bottleneck of Deep Learning - [Youtube](https://www.youtube.com/watch?v=XL07WEc2TRI)
* Maximum Entropy Distributions - [blog](http://bjlkeng.github.io/posts/maximum-entropy-distributions/)
* 



**Articles**

* A New Outlook on Shannon's Information Measures - Yeung (1991) - [pdf](https://pdfs.semanticscholar.org/a37e/ab85f532cdc027260777815d78f164eb93aa.pdf)
* A Brief Introduction to Shannon's Information Theory - Chen (2018) - [arxiv](https://arxiv.org/pdf/1612.09316.pdf)
* Information Theory for Intelligent People - DeDeo (2018) - [pdf](http://tuvalu.santafe.edu/~simon/it.pdf)

**Blogs**

* Visual Information Theory - Colah (2015) - [blog](https://colah.github.io/posts/2015-09-Visual-Information/)
* Better Intuition for Information Theory - Kirsch (2019) - [blog](https://www.blackhc.net/blog/2019/better-intuition-for-information-theory/)
* A Brief History of Information Theory - Vasiloudis  (2019) - [blog](http://tvas.me/articles/2018/04/30/Information-Theory-History.html)
* Information Theory of Deep Learning - Sharma (2019) - [blog](https://adityashrm21.github.io/Information-Theory-In-Deep-Learning/)

**Books**



* Information theory

---
## Software

### Python Packages

* Discrete Information Theory (**dit**) - [Github](https://github.com/dit/dit) | [Docs](http://dit.readthedocs.io/en/latest/index.html)
* Python Information Theory Measures (****) - [Github](https://github.com/pafoster/pyitlib/) | [Docs](https://pafoster.github.io/pyitlib/)
* Parallelized Mutual Information Measures - [blog](http://danielhomola.com/2016/01/31/mifs-parallelized-mutual-information-based-feature-selection-module/) | [Github](https://github.com/danielhomola/mifs)

### Implementations

* Mutual Information Calculation with Numpy - [stackoverflow](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)

---
## Unexplored Stuff

These are my extra notes from resources I have found.

---


