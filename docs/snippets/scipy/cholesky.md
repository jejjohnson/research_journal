# The Cholesky Decomposition


I use this quite often whenever I'm dealing with Gaussian processes and kernel methods. Instead of doing the solver, we can simply use the `cho_factor` and `cho_solve` that's built into the `scipy` library. 


## Direct Solver

```python
# use the direct solver
weights = scipy.linalg.solve(K + alpha * identity, y)
```


## Cholesky Factor

```python
# cholesky factor
L, lower = scipy.linalg.cho_factor(K + alpha * identity)

# cholesky solver
weights = scipy.linalg.cho_solve((L, lower), y)
```