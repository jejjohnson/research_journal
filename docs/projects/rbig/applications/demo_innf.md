# Demo: Gaussianization



## Data

```python

```

## RBIG Model

### Initialize Model

```python
# rbig parameters
n_layers        = 1
rotation_type   = 'PCA'
random_state    = 123
zero_tolerance  = 100
base            = 'gauss'

# initialize RBIG Class
rbig_clf = RBIG(
    n_layers=n_layers,
    rotation_type=rotation_type,
    random_state=random_state,
    zero_tolerance=zero_tolerance,
    base=base
)
```

### Fit Model to Data

```python
# run RBIG model
rbig_clf.fit(X);
```

### Visualization


#### 1. Marginal Gaussianization

```python
# rotation matrix V (N x F)
V = rbig_clf.rotation_matrix[0]

# perform rotation
data_marg_gauss = X @ V
```

#### 2. Rotation

```python

```