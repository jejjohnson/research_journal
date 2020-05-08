# Refactoring RBIG (RBIG 1.1)


## Components

### Flow

* Forward Transformation
  * Transformation
  * Log Determinant Jacobian
* Inverse Transformation
  * Transformation
  * Log Determinant Jacobian

### Normalizing Flow

This is a sequence of Normalizing Flows.

* Forward Transformation (all layers)
* Backwards Transformation (all layers)
* Output:
  * Transformation
  * Log Determinant Jacobian

### Normalizing Flow Model

This is a Normalizing flow with a prior distribution

* Init: Prior, NF Model
* Forward: Forward, LogDet, Prior
* Backward: Transform, LogDet
* Sample: Transform


---


## Ideal Case

1. Define the Prior Distribution

```python

d_dimensions = 1

# initialize prior distribution
prior = MultivariateNormal(
    mean=torch.zeros(d_dimensions),
    cov=torch.eye(d_dimensions)
)

```

2. Define the Model


```python
n_layers = 2

# make flow blocks
flows = [flow(dim=d_dimensions) for _ in range(n_layers)]

# create model given flow blocks and prior
model = NormalizingFlowModel(prior, flows)
```

3. Define Optimization scheme

```python
opt = optim.Adam(model.parameters(), lr=0.005)
```

4. Optimize Model

```python
for i in range(n_epochs):

    # initialize optimizer
    opt.zero_grad()

    # get forward transformation
    z = model.transform(x)

    # get prior probability
    prior_logprob = model.prior(x)

    # get log determinant jacobian prob
    log_det = model.logabsdet(x)

    # calculate loss
    loss = - torch.mean(prior_logprob + log_det)

    # backpropagate
    loss.backward()

    # optimize forward
    opt.step()

```

