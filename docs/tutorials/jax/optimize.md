# Optimizing Using Jax

---

### From Scratch


---

### Step Function

```python
# STEP FUNCTION
@jax.jit
def step(params, X, y, opt_state):
    # calculate loss
    loss = mll_loss(params, X, y)

    # calculate gradient of loss
    grads = dloss(params, X, y)

    # update optimizer state
    opt_state = opt_update(0, grads, opt_state)

    # update params
    params = get_params(opt_state)

    return params, opt_state, loss
```

And now we need to actually go through and initialize the parameters.

```python
# TRAINING PARARMETERS
n_epochs = 500 if not args.smoke_test else 2
learning_rate = 0.01
losses = list()

# initialize optimizer
opt_init, opt_update, get_params = optimizers.rmsprop(step_size=learning_rate)

# initialize parameters
opt_state = opt_init(params)

# get initial parameters
params = get_params(opt_state)
```

And lastly let's do the actual loop.

```python
# initialize progress bar
postfix = {}

with tqdm.trange(n_epochs) as bar:

    for i in bar:
        # 1 step - optimize function
        params, opt_state, value = step(params, X, y, opt_state)

        # store loss values
        losses.append(value.mean())

        # store parameters for display
        postfix = {}
        for ikey in params.keys():
            postfix[ikey] = f"{params[ikey]:.2f}"
        postfix["Loss"] = f"{onp.array(losses[-1]):.2f}"

        # update progress bar
        bar.set_postfix(postfix)
```


---

## Resources



#### Using Scipy Optimize

* [Scipy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) | [Minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) | [StackOverFlow](https://stackoverflow.com/questions/13670333/multiple-variables-in-scipys-optimize-minimize)
* [Scipy Lectures](https://scipy-lectures.org/advanced/mathematical_optimization/index.html)
* [Real Python - Scientific Python: Using Scipy for Optimization](https://realpython.com/python-scipy-cluster-optimize/)