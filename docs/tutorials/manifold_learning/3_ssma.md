# Semisupervised Manifold Alignment

##### Main References

* Semisupervised Manifold Alignment of Multimodal Remote Sensing Images - Tuia et al.
* Classification of Urban Multi-Angular Image Sequences by Aligning their Manifolds -
Trolliet et al.
* Multisensor Alignment of Image Manifolds - Tuia et al.
* Domain Adaption using Manifold Alignment - Trolliet


### Outline

In this chapter, I introduce the semisupervsied manifold alignment (SSMA) method that
was presented in [*linktopaper*]. I give a brief overview of the algorithm in section
1.1 followed by some notation declarations in 1.2. I proceed to talk about the cost
function in section 1.3 followed by explicit construction of some cost function components
in sections 1.4, 1.5 and 1.6. I give some insight into the projection functions in section
1.7 followed by some advantages of this method with other alignment methods in section
1.8. I conclude this chapter with some insight into the computational complexity of the
SSMA method in section 1.9.

### 1.1 Overview

The main idea of the semisupervised manifold alignment (SSMA) method is to align
individual manifolds by projecting them into a joint latent space $\mathcal{F}$. With
respect to remote sensing, the authors of [*linktopaper*] have provided evidence to
support the notion that the local geometry of each image dataset is preserved and
the regions with similar classes are brought together whilst the regions with dissimilar
classes are pushed apart in the SSMA embedding. By using graph Laplacians for each of the
similarity and dissimilarity terms, a Rayleigh Quotient function is minimized to
extract projection functions. These projection functions are used to project the
individual manifolds into a joint latent space.

#### Explicit Algorithm (*in words*)

##### 1.2 Notation

Let's have a series of $M$ images with each data matrix $X^{m}$, where $m=1,2,\ldots, M$.
Let each matrix $X^{m}$ be split into labeled samples $\{x_{i}^{m} \}^{u_{m}}_{i=1}$
and unlabeled samples $\{x_{j}^{m}, y_{j}^{m} \}^{l_{m}}_{j=1}$. Typically, there are many
more labeled samples than unlabeled samples so let's assume $l_{m} << u_{m}$. Let our data
matrix be of $d_{m}\text{x }n_{m}$ dimensions which says that $d$ dimensional data by
$n_{m}$ labeled and unlabeled samples. Succinctly, we have
$X^{m} \in \mathbb{R}^{d_{m}\text{x } n_{m}}$, where $n_{m}=l_{m}+u_{m}$. We can explicitly
write each data matrix into a block diagonal matrix $X$ where $X=diag(X_{1}, \ldots,
  X_{M})$ and $X \in \mathbb{R}^{d\text{x}N}$.

$$
X =
\begin{bmatrix}
X^1 & 0 & 0 & 0 \\
0 & X^2 & 0  & 0 \\
\vdots & \vdots & \vdots & \vdots\\
0 & \ldots & \ldots & X^M
\end{bmatrix}_{}$$

$N$ is the total number of labeled samples for all
of the images and $d$ is the overall dimension of all of the images, i.e.
$N=\sum_{m}^{M}n_{m}$ and $d=\sum_{m}^{M}d_{m}$.

*Note:* These images do not necessarily need to be exact replicas of the
the same spectral density, the same spatial dimension or even the same sensor. Practically,
this means that we do not need to have the same $m$ for each matrix $X^{m}$.

##### 1.3 Semisupervised Loss Function

In [*linktopaper*], they construct a semisupervised loss function with the idea of aligning
$M$ images to a common representation. Ultimately, they created $M$ projection functions
$\mathcal{F}$ that mapped image $X^{m}$ from $X^{m} \in \mathbb{R}^{d_{m}\text{x } n_{m}}$
to $f^{m} \in \mathbb{R}^{d_{m} \text{x }d}$ where $m=1,2, \ldots, M$. They ensure that the
projection function $f$ brings samples from the same class closer together and
pushes samples from different classes apart whilst still preserving the geometry of each
individual manifold. Abstractly, the aim is to maximize the distance between the
dissimilarities between the data sets and minimize the similarities between the datasets.
This can be expressed via the following equation:

$$\text{Cost Function}
= \frac{\text{Similarity} + \mu \text{Geometric}}{\text{Dissimilarity}}$$

As you can see, if the denominator gets very large then this cost function will get very
small.
The Rayleigh Quotient is used because the problem has terms that need
to be minimized and terms that need to be maximized.

$$F_{opt}=\underset{F}{argmin} \frac{F^TAF}{F^TBF}$$


$$F_{opt}=\underset{F}{argmin}\left\{ tr\left( (F^{T}BF)^{-1}F^{T}AF \right)\right\}$$



$F$ is a $d$x$d$ projection matrix and the row blocks of $F$ correspond to the domain
specific projection functions $f^{m} \in \mathbb{R}^{d_{m}\text{x }d}$ that project the
data matrix $X^{m}$ into the joint latent space. The matrix $A$ corresponds to the term
that needs to be minimized and the matrix $B$ corresponds to the term that needs to be
maximized. The authors break the matrix $A$ into two components, a geometric component $G$
and a class similarity component, $S$. They want to preserve the manifold of each data
matrix $X^{m}$ and bring same classes closer together in the joint latent subspace
manifold. They combine the geometric component, $G$ and the similarity component, $S$ to
create the discriminant component $A$, i.e. $A=S+\mu G$ where $\mu$ is a free parameter to
monitor the tradeoff between the local manifold and the class similarity. The authors
choose to use three graph Laplacian matrices to construct the terms in the Rayleigh
quotient, $G$, $S$, and $B$.


##### 1.4 Geometric Spectral Similarity Term

The $G$ term preserves the manifold of each individual data set $X^{m}$ throughout the
transformation as no inter-domain relationships are considered. Affinity matrices are
constructed for each $X^{m}$ and then put into a
block diagonal matrix $W^{m}_{g} \in \mathbb{R}^{n_{m}\text{x }n_{m}}$. There are many
ways to construct affinity matrices including $k$ nearest neighbour graphs ($k$-NN),
$\epsilon$-neighbourhood graphs ($\epsilon$-N) and Gaussian graphs. The graphs are assumed
to be undirected so the affinity matrix should be symmetric, i.e. if $x_{i}$ is connected
to $x_{j}$ then $x_{j}$ is connected to $x_{i}$. An example of the final matrix $W_g$ is
like so:

$$
W_g =
\begin{bmatrix}
W_g^1 & 0 & 0 & 0 \\
0 & W_g^2 & 0  & 0 \\
\vdots & \vdots & \vdots & \vdots\\
0 & \ldots & \ldots & W_g^M
\end{bmatrix}$$

Entries in the matrices $W_g^m$ are $W_{g}^{m}(i,j)=1$ if $x_{i}$ and $x_{j}$ are
connected and 0 otherwise.
Using the $k$-NN weighted graph, this term ensures that the samples in the
original $m$ domain remains in the same proximity in the new latent projected space.
A graph laplacian, $L_{g}^{m}$ is constructed from each of the affinity matrices and put
into a block diagonal matrix $L_g$, where $L_g^m=D_{g}^{m}-W_{g}^{m}$ and $D_{g}^{m}$ is
the degree matrix defined as $D_g^m(i,i)=\sum_{j}^{n_{m}}W_g^m(i,j)$. The laplacian matrix
$L$ is positive semi-definite and symmetric. An example of the final matrix $D_g$ and
$L_g$ are below:

$$
D_g =
\begin{bmatrix}
D_g^1 & 0 & 0 & 0 \\
0 & D_g^2 & 0  & 0 \\
\vdots & \vdots & \vdots & \vdots\\
0 & \ldots & \ldots & D_g^M
\end{bmatrix}$$

which just simplifies to

$$
L_g = D_g - W_g =
\begin{bmatrix}
L_g^1 & 0 & 0 & 0 \\
0 & L_g^2 & 0  & 0 \\
\vdots & \vdots & \vdots & \vdots\\
0 & \ldots & \ldots & L_g^M
\end{bmatrix}$$

The term that needs to be minimized is now:

$$
G=
\sum_{m=1}^{M}\sum_{i,j=1}^{n_{m}}W_{g}^{m}(i,j)||f^{mT}x_{i}^{m}-f^{mT}x_{j}^{m}||^{2}
$$
$$
G = tr(F^T X L_g X^T F)
$$

(*Put a note about the graph Laplacian. Maybe an appendix section?*)

##### 1.5 Class Label Similarity Term

The $S$ term ensures that the inter-domain classes are brought together in the joint
latent space by pulling labeled samples together. The author use a matrix of class
similarities to act as an affinity matrix. Entries in the matrices $W_s^{m,m'}(i,j)=1$ if
the samples between class $m$ and $m'$ have the same label and 0 otherwise where
$m, m'=1,2,\ldots, M$.

Because this affinity matrix only comparing the labeled samples, we can be sure that this
matrix will be very sparse.

A graph laplacian, $L_s$ is constructed in the same way as the geometric term is.
The term that needs to be minimized is now:

$$
S=
\sum_{m,m'=1}^{M}\sum_{i,j=1}^{l_m,l_m'}W_s^{m,m'}(i,j)||f^{mT}x_i^m-f^{mT}x_j^m||^{2}
$$
$$
S = tr(F^T X L_s X^T F)
$$

##### 1.6 Class Label Dissimilarity Term

The $B$ term ensures that the inter-domain classes are pushed apart in the joint
latent space by pushing different labeled samples apart. The author use a matrix of class
dissimilarities to act as an affinity matrix. Entries in the matrices $W_d^{m,m'}(i,j)=1$
if the samples between class $m$ and $m'$ have different labels and 0 otherwise, where
$m, m'=1,2,\ldots, M$.

Because this affinity matrix is only comparing the labeled samples, we can be sure that
this matrix will be very sparse.

A graph laplacian, $L_d$ is constructed in the same way as the geometric term is.
The term that needs to be minimized is now:

$$
B=
\sum_{m,m'=1}^{M}\sum_{i,j=1}^{l_m,l_m'}W_d^{m,m'}(i,j)||f^{mT}x_i^m-f^{mT}x_j^m||^{2}
$$

which again simplifies to
$$
B = tr(F^T X L_d X^T F)
$$

##### 1.7 Projection Functions

We can aggregate all of the graph Laplacians ($L_g, L_s, L_d \in \mathbb{R}^{n\text{x}n}$)
and then use the Rayleigh Quoitient formulation to get our cost function that needs to
be minimized:

$$F_{opt}=\underset{F}{argmin}\left\{ tr\left(
  (F^{T}XL_dX^TF)^{-1}
  F^{T}X(\mu L_g+L_s)X^TF
  \right)\right\}$$

*(I want to talk more about the constraints...)*

We can find a solution to this minimization problem by looking for the smallest eigenvalues
$\gamma_i$ of the generalized eigenvalue problem:

$$X(\mu L_g + L_s)X^T \Gamma = \lambda XL_dX^T\Gamma$$

The optimal solution to the generalized eigenvalue problem contains the projection
functions necessary to project each $X^m$ into the joint latent space. If we look at
$F_{opt}$, we can see the matrix will be built as follows:

$$F_{opt}= \left[ \sqrt{\lambda_1}\gamma_{1}|\ldots|\sqrt{\lambda_M}\gamma_{M}\right]$$
$$F_{opt}=
\begin{bmatrix}
f_1^1 & \ldots & f_d^1 \\
f_1^2 & \ldots & f_d^2 \\
\vdots & \vdots & \vdots \\
f_1^M & \ldots & f_d^M
\end{bmatrix}$$

To put it succinctly, we have a projection of $X^m$ from domain $m$ to the joint
latent space $F$ of dimension, $d$:

$$\mathcal{P}_{f}(X^m)=f^{mT}X^m$$

Furthermore, any data matrix $X^m$ can be projected into the latent space of data matrix
$X^{m'}$. Let $X^p=X^mf^m$ be the projection of data matrix $X^m$ into the joint latent
space. This can be translated into the latent space of $X^{m'}$ via the translation
$X^{m1} =X^mf^m(f^{m'})^\dagger$ where $(f^{m'})^\dagger$ is the pseudoinverse of the
eigenvectors of domain $m'$.


### 1.8 Properties and Comparison to other methods

The authors outline some significant properties that set asside the SSMA method from
other methods that have been used for domain adaption.

##### Linearity

This method constructs and defines explicit projection functions that can project data
matrices $X^m$ into the joint latent space. That being said, this means that the

##### Multisensor

This method only considers the geometry of each individual $X^m$ exclusively and so there
is no restriction on the number of hyperspectral image bands nor on the spectral band
properties or amount.

##### Multidomain

This method can align domains to a joint latent space so there is no limitation on the
number of domains. There is no requirement for there to be a leading domain where all other
domains are similar to; although it would be wise to do so.


##### PDF-based

This method aligns the underlying manifold for each domain and so there is no need for
there to be an co-registration of source domains nor does there need to be images with
the same spatial or spectral resolution.

##### Invertibility

This method creates explicit projection functions that map from one target domain to
a joint latent space. Likewise, it is also possible to project data from one target domain
to another target domain.


### 1.9 Computational Complexity

The SSMA method is a typical graph-based computational problem.

The bulk of the computational effort comes with constructing the graph Laplacians,
specifically the geometric term. A $k-NN$ method is used to create the affinity matrix
which can be on order (*order?*). The similarity and dissimilary terms are constructed
by convolving vectors with the class labels which results in very sparse matrices.

Storing matrices can be memory intense but there exists methods to combat this limitation.
*(Need to go into methods which can reduce the storage for large matrices.)*

There can be a significant cost to compute the eigenvalues of the generalized eigenvalue
problem which can be of order (*order?*). We can combat the cost by either reducing the
size of the problem or improving the algebraic computational process. Some iterative
methods can be used to reduce the computational complexity from $\mathcal{O}(d^3)$ to
$\mathcal{O}log(d)$.
The cost to compute the eigenvalue decomposition can be avoided using faster methods such
as approximate random projection singular value decomposition (ARSVD), Jacobi-Davidson
QR (JDQR) factorization method, or multigrid methods.
