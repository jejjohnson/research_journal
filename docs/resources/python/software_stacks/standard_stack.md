---
title: Standard
description: The standard python stack
authors:
    - J. Emmanuel Johnson
path: docs/resources/python/software_stacks
source: standard_stack.md
---

# Standard Python Stack

For the most part, you'll see this at the top of everyone's scientific computing notebook/script:

```python

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

```

This is the impact of these libraries on the python community. By far, the most mature, the most robust and have the most documentation. In principle, you should be able to do almost any kind of scientific computing from start to finish with these libraries. Below I list them and I give the most useful resources that I have use (and still use). Just remember, when in doubt, stackoverflow is your best friend.

---

### **Numpy**

The standard library for utilizing the most fundamental data structure for scientific computing: the array. It also has many linear algebra routines that have been optimized in C/C++ behind the scences. Often times doing things in raw python can get a massive speed up by doing things with numpy. The must have package for everyones python stack. It also has some of the best documentation for python packages.

??? info "Tutorials"
    * [Documentation](https://numpy.org/devdocs/user/quickstart.html)
    * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) - Jake Vanderplas - Chapter 2 - Intro to Numpy
    * [Intro to Numpy](https://www.youtube.com/watch?v=ZB7BZMhfPgk)
    * [A Visual Guide to Numpy](https://jalammar.github.io/visual-numpy/)
    * [From Python to Numpy](http://www.labri.fr/perso/nrougier/from-python-to-numpy/)
    * [100 Exercises in Numpy](https://github.com/rougier/numpy-100)
    * [Broadcasting Tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)
    *  Einsum - [I](https://rockt.github.io/2018/04/30/einsum) | [II](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)



---

### **Scipy**

The other defacto library for doing scientific computing. This package is quite heavily linked with numpy but it does have it's own suite of routines. This library also has linear algebra routines including sparse matrices. But it also has signal processing, probability and statistics and optimization functions. Another library with some of the best documentation.

??? info "Resources"
    * [Documentation](https://docs.scipy.org/doc/scipy/reference/)
    * [Scipy Lecture Notes](https://scipy-lectures.org/)


---

### **Scikit-Learn**


This **is the** de facto library for machine learning. It will have all of the standard algorithms for machine learning. They also have a lot of utilities that can be useful when preprocessing or training your algorithms. The API (the famous `.fit()`, `.predict()`, and `.transform()`) is great and has been adopted in many other machine learning packages. There is also no other tutorial that comes close to being helpful as the documentation.

??? info "Resources"
    * [Documentation](https://scikit-learn.org/stable/)
    * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html) - Jake Vanderplas - Chapter 5 - Machine Learning
    * [Creating your own estimator in scikit-learn](http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/) - Daniel Hnyk (2015)


---

### **Pandas**

The also most fundamental package that's often used in the preprocessing chain is the fundamental datastructure known as a pandas table. Taking inspiration from R, this features data with meaningful meta data attached to it as columns or as rows. It also allows entries other than floats and ints. The best part is that is has reallt fast routines to do data manipulation and advanced calculations. This is the hardest package to get accustomed to compared to the previous packages but with a little time and effort, you can easily become one of the most effective data scientists around. The documentation is excellent but I don't find it as friendly as the other packages. Because it has a bit of a learning curve and there are likely more than 5 ways to do what you want to do, you'll find a lot of tutorials online.

??? info "Resources"
    * [Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html)
    * [Chris Albon Snippets](https://chrisalbon.com/)
    * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html) - Jake Vanderplas - Chapter 3 - Intro to Pandas
    * Greg Reda 3 part Tutorial
      * I - [Intro to Pandas Structures](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/)
      * II - [Working with DataFrames](http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/)
      * III - [Using Pandas with the MovieLens Dataset](http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/)
    * Tom Augspurger - 7-Part Tutorial - [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)
    * Chris Fonnesneck Class - [Advanced Statistical Computing Class](https://github.com/fonnesbeck/Bios8366)


---

### **Matplotlib**

This is the default plotting library for python. Most people have a love-hate relationship with this library. I find it very easy to plot things but it gets difficult when I need to modify something down to the T. Like pandas, I think this library has a bit of a learning curve if you really want to make kick-ass plots. The documentation is good but the problem is I don't understand the data structures or language for creating plots; creating plotting libraries is quite difficult for this very reason. This is a great library but prepared to be a bit frustrated at times down the road.


??? info "Resources"
    * [Matplotlib Gallery](https://matplotlib.org/gallery/index.html)
    * [Anatomy of Matplotlib](https://github.com/matplotlib/AnatomyOfMatplotlib)
    * [An Inquiry Into Matplotlib's Figures](https://matplotlib.org/matplotblog/posts/an-inquiry-into-matplotlib-figures/)
    * [Python Plotting with Matplotlib](https://realpython.com/python-matplotlib-guide/) - Real Python
    * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) - Jake Vanderplas - Chapter 4 - Visualization with Matplotlib
    * [Creating Publication-Quality Figures with Matplotlib](https://github.com/jbmouret/matplotlib_for_papers)
    * [Matplotlib Cheatsheet](https://github.com/rougier/matplotlib-cheatsheet)