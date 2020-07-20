# Kernel Derivatives

## Linear Kernel


---

## RBF Kernel


$$
k(x,y) = \exp(-\gamma ||x-y||_2^2)
$$

---

### 1st Derivative

We can calculate the cross-covariance term $K_{fg}(\mathbf{x,x})$. We apply the following operation

$$
K_{fg}(x,x') = k_{ff}(\mathbf{x,x'})(1, \frac{\partial}{\partial x'})
$$
If we multiply the terms across, we get:
$$
K_{fg}(x,x') = k_{ff}(\mathbf{x,x'})\frac{\partial k_{ff}(\mathbf{x,x'})}{\partial x'}
$$

For the RBF Kernel, it's this:

$$\frac{\partial k(x,y)}{\partial x^j}=-2 \gamma (x^j - y^j) k(x,y)$$

---

### 2. Cross-Covariance Term - 2nd Derivative

Recall the 1st derivative is:

$$\frac{\partial k(x,y)}{\partial x^j}=-2 \gamma (x^j - y^j) k(x,y)$$

So now we repeat. First we decompose the function using the product rule:


$$
\begin{aligned}
\frac{\partial^2 k(x,y)}{\partial x^{j^2}} &=
-2 \gamma (x^j - y^j) \frac{\partial }{\partial x^j} k(x,y) + k(x,y) \frac{\partial }{\partial x^j} \left[ -2 \gamma (x^j - y^j) \right]\\
\end{aligned}
$$

The first term is basically the 1st Derivative squared and the 2nd term is a constant. So after applying the derivative and simplifying, we get:

$$
\begin{aligned}
\frac{\partial^2 k(x,y)}{\partial x^{j^2}} &=
4 \gamma^2 (x^j - y^j)^2 k(x,y) -2 \gamma k(x,y)\\
&=
\left[ 4\gamma^2(x^j - y^j)^2 - 2\gamma\right] k(\mathbf{x}, \mathbf{y}) \\
&=
2 \gamma \left[ 2\gamma(x^j - y^j)^2 - 1\right] k(\mathbf{x}, \mathbf{y}) \\
\end{aligned}
$$

---

### 3. Cross-Covariance Term - 2nd Derivative (Partial Derivatives)

Recall the 1st derivative is:

$$\frac{\partial k(x,y)}{\partial x^j}=-2 \gamma (x^j - y^j) k(x,y)$$

So now we repeat. First we decompose the function using the product rule. But this time, we need to do the product rule first w.r.t. $x^j$ and then w.r.t. $y^k$.

$$
\begin{aligned}
\frac{\partial^2 k(x,y)}{\partial x^j y^k} &=
-2 \gamma (x^j - y^j) \frac{\partial }{\partial y^k} k(x,y) + k(x,y) \frac{\partial }{\partial y^k} \left[ -2 \gamma (x^j - y^j) \right]\\
\end{aligned}
$$

So now let's start expanding and collapsing terms:

$$
\begin{aligned}
\frac{\partial^2 k(x,y)}{\partial x^j y^k} &=
4 \gamma^2 (x^j - y^j)(x^k - y^k) k(x,y) \\
\end{aligned}
$$

The second term should go to zero and the first term is the same except it has different dimensions (w.r.t. $y$ instead of $x$).

$$
\frac{\partial^2 k(x,y)}{\partial x^j \partial y^k} =
4 \gamma^2 (x^k - y^k)(x^j - y^j) k(\mathbf{x}, \mathbf{y})
$$