---
## KL Divergence


**Typical**:

$$\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x)\right] =
\int_\mathcal{X} q(\mathbf x) \log \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} d\mathbf x$$

**VI**:

$$\text{KL}\left[ q(\mathbf x) || \mathcal{P}(\mathbf x)\right] = -
\int_\mathcal{X} q(\mathbf x) \log \frac{\mathcal{P}(\mathbf x)}{q(\mathbf x)} d\mathbf x$$

#### Positive and Reverse KL



* Density Ratio Estimation for KL Divergence Minimization between Implicit Distributions - [Blog](https://tiao.io/post/density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions/)
**Resources**:
* YouTube
  * Aurelien Geron - [Short Intro to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
  * Ben Lambert - [Through Secret Codes](https://www.youtube.com/watch?v=LJwtEaP2xKA)
  * Zhoubin - [Video](https://youtu.be/5KdWhDpeQvU)
    > A nice talk where he highlights the asymptotic conditions for MLE. The proof is sketched using the minimization of the KLD function.
* Blog
  * [Anna-Lena Popkes](https://github.com/zotroneneis/resources/blob/master/KL_divergence.ipynb)
  * [KLD Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
  * [KLD for ML](https://dibyaghosh.com/blog/probability/kldivergence.html)
  * [Reverse Vs Forward KL](http://www.tuananhle.co.uk/notes/reverse-forward-kl.html)
  * [KL-Divergence as an Objective Function](https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/)
  * [NF Slides (MLE context)](https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-09-Introduction-to-Normalizing-Flows/slides.pdf)
  * [Edward](http://edwardlib.org/tutorials/klqp)
* Class Notes
  * Stanford - [MLE](https://web.stanford.edu/class/stats200/Lecture13.pdf) | [Consistency and Asymptotic Normality of the MLE](https://web.stanford.edu/class/stats200/Lecture14.pdf) | [Fisher Information, Cramer-Raw LB](https://web.stanford.edu/class/stats200/Lecture15.pdf) | [MLE Model Mispecification](https://web.stanford.edu/class/stats200/Lecture16.pdf)
* Code
  * [KLD py](https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7)
  * [NumPy/SciPy Recipes](https://www.researchgate.net/publication/278158089_NumPy_SciPy_Recipes_for_Data_Science_Computing_the_Kullback-Leibler_Divergence_between_Generalized_Gamma_Distributions)