# Linear Algebra Tricks


---

- [Frobenius Norm (Hilbert-Schmidt Norm)](#frobenius-norm-hilbert-schmidt-norm)
  - [Intiution](#intiution)
  - [Formulation](#formulation)
  - [Code](#code)
- [Frobenius Norm](#frobenius-norm)
  - [Frobenius Norm (or Hilbert-Schmidt Norm) a matrix](#frobenius-norm-or-hilbert-schmidt-norm-a-matrix)

---

## Frobenius Norm (Hilbert-Schmidt Norm)

### Intiution

The Frobenius norm is the common matrix-based norm.

---

### Formulation

$$
\begin{aligned}
||A||_F &= \sqrt{\langle A, A \rangle_F} \\
||A|| &= \sqrt{\sum_{i,j}|a_{ij}|^2} \\
&= \sqrt{\text{tr}(A^\top A)} \\
&= \sqrt{\sum_{i=1}\lambda_i^2}
\end{aligned}$$


<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

Let $A=U\Sigma V^\top$ be the Singular Value Decomposition of A. Then

$$||A||_{F}^2 = ||\Sigma||_F^2 = \sum_{i=1}^r \lambda_i^2$$

If $\lambda_i^2$ are the eigenvalues of $AA^\top$ and $A^\top A$, then we can show 

$$
\begin{aligned}
||A||_F^2 &= tr(AA^\top) \\
&= tr(U\Lambda V^\top V\Lambda^\top U^\top) \\
&= tr(\Lambda \Lambda^\top U^\top U) \\
&= tr(\Lambda \Lambda^\top) \\
&= \sum_{i}\lambda_i^2
\end{aligned}
$$

</details>

---

### Code

**Eigenvalues**

```python
sigma_xy = covariance(X, Y)
eigvals = np.linalg.eigvals(sigma_xy)
f_norm = np.sum(eigvals ** 2)
```

**Trace**

```python
sigma_xy = covariance(X, Y)
f_norm = np.trace(X @ X.T) ** 2
```

**Einsum**

```python
X -= np.mean(X, axis=1)
Y -= np.mean(Y, axis=1)
f_norm = np.einsum('ij,ji->', X @ X.T)
```

**Refactor**

```python
f_norm = np.linalg.norm(X @ X.T)
```

---

## Frobenius Norm

$$||X + Y||^2_F = ||X||_F^2 + ||Y||_F^2 + 2 \langle X, Y \rangle_F$$


### Frobenius Norm (or Hilbert-Schmidt Norm) a matrix

$$
\begin{aligned}
||A|| &= \sqrt{\sum_{i,j}|a_{ij}|^2} \\
&= \sqrt{\text{tr}(A^\top A)} \\
&= \sqrt{\sum_{i=1}\lambda_i^2}
\end{aligned}$$


<!-- <details> -->
<summary>
    <font color="black">Details
    </font>
</summary>

Let $A=U\Sigma V^\top$ be the Singular Value Decomposition of A. Then

$$||A||_{F}^2 = ||\Sigma||_F^2 = \sum_{i=1}^r \lambda_i^2$$

If $\lambda_i^2$ are the eigenvalues of $AA^\top$ and $A^\top A$, then we can show 

$$
\begin{aligned}
||A||_F^2 &= tr(AA^\top) \\
&= tr(U\Lambda V^\top V\Lambda^\top U^\top) \\
&= tr(\Lambda \Lambda^\top U^\top U) \\
&= tr(\Lambda \Lambda^\top) \\
&= \sum_{i}\lambda_i^2
\end{aligned}
$$
