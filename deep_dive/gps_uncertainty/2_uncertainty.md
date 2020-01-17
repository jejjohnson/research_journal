# Input Uncertainty

## Intuition: 1D Regression



**Sources**

* Intution Examples - [Colab Notebook](https://colab.research.google.com/drive/1uXY0BqHIXlymj9_I0L0J8S-iaa8Pnf0B)



### Real Function

<p>
  <img src="pics/egp_real.png" alt="drawing" width="400"/>
</p>

### Output Uncertainty



<p>
  <img src="pics/egp_ey.png" alt="drawing" width="400"/>
</p>

### Input Uncertainty

<p align="center">
  <img src="pics/egp_ex.png" alt="drawing" width="400"/>
</p>

### Input and Output Uncertainty

<p>
  <img src="pics/egp_exy.png" alt="drawing" width="400"/>
</p>



### An Illusion: Noisy Inputs through function

<p>
  <img src="pics/egp_efy.png" alt="drawing" width="400"/>
</p>

### Intuition: Confidence Intervals

<p align="center">
  <img src="pics/vertical_errors.png" alt="drawing" width="800"/>
</p>

**Figure**: Intuition of for the Taylor expansion for a model:

* a) $y=f(\mathbf x) + \epsilon_y$
* b) $y=f(\mathbf x + \epsilon_x)$
* c) $y=f(\mathbf x + \epsilon_x) + \epsilon_y$

The key idea to think about is what contributes to how far away the error bars are from the approximated mean function. The above graphs will help facilitate the argument given below. There are two main components:

1. **Output noise $\epsilon_y$** - the further away the output points are from the approximated function will contribute to the confidence intervals. However, this will affect the vertical components where it is flat and not so much when there is a large slope.
2. **Input noise $\epsilon_x$** - the influence of the input noise depends on the slope of the function. i.e. if the function is fully flat, then the input noise doesn't affect the vertical distance between our measurement point and the approximated function; contrast this with a function fully sloped then we have a high contribution to the confidence interval.

So there are two components of competing forces: $\sigma_y^2$ and $\epsilon_x$ and the $\epsilon_x$ is dependent upon the slope of our function $\partial f(\cdot)$ w.r.t. $\mathbf x$. 

