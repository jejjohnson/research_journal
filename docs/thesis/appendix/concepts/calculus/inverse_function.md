# Inverse Function Theorem


**Resources**:
* [Wiki](https://en.wikipedia.org/wiki/Inverse_function_theorem)
* YouTube
  * Prof Ghist Math - [Inverse Function Theorem](https://www.youtube.com/watch?v=LWk7hvY1Goc)
  * The Infinite Looper - [Inv Fun Theorem](https://www.youtube.com/watch?v=gS0TYC78lnw&t=186s)
  * Professor Leonard - [Fundamental Theorem of Calculus](https://www.youtube.com/watch?v=xjtEfS0vY2o&list=PLF797E961509B4EB5&index=29&t=0s) | [Derivatives of Inverse Functions](https://www.youtube.com/watch?v=HnsUNWNYZ28)


**Source**: 

* [Mathematics for Machine Learning](https://mml-book.github.io/book/chapter06.pdf) - Deisenroth (2019)
* [Change of Variables: A Precursor to Normalizing Flow](http://ruishu.io/2018/05/19/change-of-variables/) - Rui Shu
* [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) - Bishop (2006)

Often we are faced with the situation where we do not know the distribution of our data. But perhaps we know the distribution of a transformation of our data, e.g. if we know that $X$ is a r.v. that is uniformly distributed, then what is the distribution of $X^2 + X + c$? In this case, we want to understand what's the relationship between the distribution we know and the transformed distribution. One way to do so is to use the inverse transform theorem which directly uses the cumulative distribution function (CDF).

Let's say we have $u \sim \mathcal U(0,1)$ and some invertible function $f(\cdot)$ that maps $X \sim \mathcal P$ to $u$.

$$x = f(u)$$

Now, we want to know the probability of $x$ when all we know is the probability of $u$. 

$$\mathcal P(x)=\mathcal P(f(u)=x)$$

So solving for $u$ in that equation gives us:

$$\mathcal P(x) = \mathcal P(u=f^{-1}(x))$$

Now we see that $u=f^{-1}(x)$ which gives us a direct formulation for moving from the uniform distribution space $\mathcal U$ to a different probability distribution space $\mathcal P$.

**Probability Integral Transform**

**Resources**
* [Brilliant](https://brilliant.org/wiki/inverse-transform-sampling/)
  > Does a nice example where they talk about the problems with fat-tailed distributions.
* [Wiki](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
* CrossValidated
  * [How does the inverse transform method work](https://stats.stackexchange.com/questions/184325/how-does-the-inverse-transform-method-work)
  * [Help me understand the quantile (inverse CDF) function](https://stats.stackexchange.com/questions/212813/help-me-understand-the-quantile-inverse-cdf-function)
* Youtube
  * Ben Hambert - [Intro to Inv Transform Sampling](https://www.youtube.com/watch?v=rnBbYsysPaU)
  * Mathematical Monk
    * [Intro](https://www.youtube.com/watch?v=rnBbYsysPaU&t=1s) | [General Case](https://www.youtube.com/watch?v=S7EXgOomvgc) | [Invertible Case](https://www.youtube.com/watch?v=irheiVXJRm8)
* Code Review - [Inverse Transform Sampling](https://codereview.stackexchange.com/questions/196286/inverse-transform-sampling)
* R Markdown - [Inverse Transform Sampling](https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html)
* Using Chebyshev - [Blog](http://www.pwills.com/blog/posts/2018/06/24/sampling.html) | [Code](https://github.com/peterewills/itsample)
* **CDFs** - Super powerful way to visualize data and also is uniformly distriuted
  * Histograms and CDFs - [blog](https://iandzy.com/histograms-cumulative-distribution/)
  * Why We Love CDFS so Much and not histograms - [Blog](https://www.andata.at/en/software-blog-reader/why-we-love-the-cdf-and-do-not-like-histograms-that-much.html)
* **Boundary Issues**
  * Confidence Band from DKW inequality - [code](http://www.statsmodels.org/devel/_modules/statsmodels/distributions/empirical_distribution.html#_conf_set)
  * Make Monotonic - [code](https://stackoverflow.com/questions/28563711/make-a-numpy-array-monotonic-without-a-python-loop)
  * Matplotlib example of CDF bins versus theoretical (not smooth) - [Code](https://matplotlib.org/gallery/statistics/histogram_cumulative.html)
* **Alternatives**
  * *KDE*
    * Statsmodels Implementation - [code](http://www.statsmodels.org/devel/_modules/statsmodels/nonparametric/kde.html) | [Univariate](https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kde.KDEUnivariate.html)
    * KDE vs Histograms - [blog](https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28)
  * *Empirical CDF*
    * The Empirical Distribution Function - [blog](http://bjlkeng.github.io/posts/the-empirical-distribution-function/)
    * Plotting an Empirical CDF In python - [blog](https://www.jddata22.com/home//plotting-an-empirical-cdf-in-python)
    * Scipy histogram - [code](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html#scipy.stats.rv_histogram)
    * Empirical CDF Function - [code](http://www.statsmodels.org/devel/_modules/statsmodels/distributions/empirical_distribution.html#ECDF)
    * ECDFs - [notebook](https://github.com/ericmjl/bayesian-analysis-recipes/blob/master/notebooks/ecdfs.ipynb)


## Derivative of an Inverse Function

* MathInsight - [Link](https://mathinsight.org/derivative_inverse_function)