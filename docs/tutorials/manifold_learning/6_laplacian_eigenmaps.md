
### Laplacian Eigenmaps

Given $m$ points $\left\{ x_1, x_2, \ldots, x_m \right\} \in \mathcal{R}^N$, we assume
the $m$ points belong to an $n$ dimensional manifold where $n$ is much smaller or equal
to $N$. Our objective is to find a lower dimensional representation
$\left\{ y_1, y_2, \ldots, y_m \right\} \in \mathcal{R}^n$ where $n<<N$.

##### Step 1. Construct the Adjacency Matrix

We build a graph $G$ whose nodes $i$ and $j$ are connected if $x_i$ is among $k$-NN or
$\epsilon$-ball graph. The distances between the data points are measured in the
euclidean distance metric. This adjacency matrix $A$ represents the connectivity of the
original data where $k$, the number of neighbours, is the free parameter.

##### Step 2. Use the heat kernel as weights for the Adjacency Matrix

We want weighted edges on the graph to represent the proximity of the vertices to their
adjacent neighbours. One common kernel to use is the diffusion weight matrix, $W$.
We can define $W$ like so:

$$W_{ij} =
\left\{
	\begin{array}{ll}
		e^{-\frac{||x_i-x_j||^2_2}{2\sigma^2}}  & \text{if } i,j \text{ are connected}\\
		0 & \text{otherwise.}
	\end{array}
\right.
$$


##### Step 3. Solve the eigenvalue problem

Let $D$ be a diagonal matrix $D_{ii}= \sum_{j}W_ij$. We can denote the lower dimensional
representation by an $m$x$n$ matrix $y=(y_1, y_2, \ldots, y_m)^T$ where each row vector
$y_i \in \mathcal{R}^n$. Now, we want to minimize the following cost function

$$\underset{y^TDy=I}{\text{min }} \frac{1}{2}\sum_{i,j}||y_i-y_j||^2W_{ij}$$

which is equivalent to minimizing the following

$$\underset{y^TDy=I}{\text{min }} \text{tr}\left( y^TLy \right)$$

where $L=D-W$ is an $m$x$m$ laplacian operator and $I$ is the identity matrix.

The constraint $y^TDy=I$ denotes...


The solution to this minimization problem is given by finding the first $n$ eigenvalue
solutions to the generalized eigenvalue problem:
$$Lf=\lambda Df$$




##### Step 4. Normalized Eigenvalue Problem (optional)

If the graph is fully connected, then $\mathbf{1}=(1,1, \ldots, 1)^T$ is the only
eigenvector with eigenvalue 0.

Instead of the above generalized eigenvalue problem, we can solve for the

$$\text{min tr}(y^TLy)$$

subject to $y^TDy=I$ and $y^TD^{\frac{1}{2}}y=0$. We can apply the $z=D^{\frac{1}{2}}y$
transformation to yield the following eigenvalue problem:

$$\text{min tr}(z^T\mathcal{L}z)$$

subject to the constraints $z^Tz=I$ and $z^T\mathbf{1}=0$ where
$\mathcal{L}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$.

$$\mathcal{L}f=\lambda f$$

(*Need some help understanding the significance of the normalized laplacian.*)


---
