# VMAP

This is a really nifty function that allows you to automatically handle batch dimensions without any cost! It allows you to write your code in a vectorized format and then just use this transformation to handle the batching.


On the website, they have an example if you have a linear function written in vector format, and you want to make predictions for multiple samples. So an example function is:

```python
def predict(params, input_vector):
    w, b = params
    output_vec = np.dot(w, input_vector) + b
    output_vec = np.tanh(output_vec)
    return output_vec
```

The thing to notice is that the input vector is on the **right-hand side** of the matrix multiplication. Normally, when we want to handle samples, we do `np.dot(inputs, W)` because we are using matrix multiplication to handle the samples. So mathematically, a vector would be:

$$
y = \mathbf{w}\mathbf{x} + b
$$

where $\mathbf{x,w} \in \mathbb{R}^{D}$ are vectors and $y,b \in \mathbb{R}$ are scalars. In almost **every machine learning tutorial [ever](https://pytorch.org/tutorials/beginner/nn_tutorial.html#neural-net-from-scratch-no-torch-nn)**, they teach you to do change that for the following:

```python
output_vec = np.dot(w, input_vector) + b
```

because it is basically implied that we've collected all of our data points into a matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ where every row is a sample and every column is a feature. It is equivalent to the notation:

$$
y = \mathbf{Xw} + b
$$


So we don't have to worry about this step. Well, pivoting off of the vector notation, if we wanted to actually account for samples, then we would need to run our `predict` function multiple times which (in python) is inefficient and it would take ages. They provide an example of how one could do it using pure functional python.

```python
from functools import partial
predictions = np.stack(list(map(partial(predict, params), input_batch)))
```

This is a pure functional programming as it features a composition of functions. I'm not well versed in functional programming and I had to stare at this for a long time before I understood. So I decided to break this down bit by bit.

### 1. "Partial" fit function the predict function

```python
predict_f = partial(predict, params)
```

We fix the params variable in the predict function by "partially fitting" the function. We are essentially fixing the params variable in the predict function and then reusing this function later. So our function went from `predict(param, X)` to `predict_(X)` using the partial function.

**Note**: Why not use the `lambda` function? This can have problems if for some reason you alter the parameters argument. The lambda still leaves the  variable open. Example: if I say `params=1` and then I use partial fit, that param will always be `1` for the `predict_` function. If I use the following:

```python
predict_l = lambda x: predict(params, x)
```

now I can change the params to be `2` and this will be reflected in the function. It depends on the application but I would avoid lambda functions when it comes to using `vmap` as you can easily get recursion issues. Make a decision, fix the function, then batch process it.

### 2. "Map" batches through new function

```
preds = list(map(predict_f, input_batch))
```

Now we decide to map all of the batch inputs through the predict_f. It's like doing a list comprehension:

```
preds = [predict_f(ix) for ix in input_batch]
```

but it's most succinct. Now our output is a list of all of the predictions.

### 3. "stack" predictions to create an array of outputs

```python
preds = np.stack(preds)
```

Now we can create the output array but stacking all of the entries on top of each other. We started off with an array of batches, and we looped through all of the inputs. The output from the map function is a list of all batches. And to get an array, we need to stack or concatenate the entries. 

**Note**: For a list of 2D arrays, we need to use the `np.vstack` or `np.hstack` depending on if we want to stack arrays via the samples or via the features.

### "vmap" batch inputs jax style

```python
# create a batch function
predict_batch_f = jax.vmap(partial(predict, params))

# profit, free inputs with batches
predictions = jax.predict_batch_f(input_batch)
```

Jax has this convenience function `vmap` which essentially vectorizes your operations. This is very similar to the numpy `vectorize` function. However, the numpy implementation isn't for performance whereas the jax `vmap` function "just works" at apparently no extra cost. We've essentially collapsed the `stack` and `map` operations into one. So now we can use our predict function above which was created in vector format, to handle batches, essentially for free!


#### More Explicit Version

```python
# create a batch function
predict_batch_f = jax.vmap(predict, in_axes=(None, 0))

# profit, free inputs with batches
predictions = predict_batch_f(params, input_batch)
```

Personally, I find this one more readable than the standard Jax implementation. Notice how we don't have to use the `partial` function because we have specified which arguments are to be vectorized and which no. I think this is better for 3 reasons:

* It's more readable. I understand exactly which arguments have batch dimensions and which do not. The other makes assumptions that only the first input (**after** partially fitting your function) is the one with batch dimensions.
* If you're function doesn't have too many arguments, then this is a bit cleaner in my opinion. If not, it may be best to fix a number of arguments in your function first to reduce the clutter.
* You cannot partially fit arguments in whichever order you please. For example, I cannot partially fit the input_batch and leave the params argument free. I don't know why this is the case. There are many tricks to get around this like using a [closure](https://stackoverflow.com/questions/11173660/can-one-partially-apply-the-second-argument-of-a-function-that-takes-no-keyword) but I think it's just easier to use the `in_axes` arguments and just be explicit.

### Why do you needs this?

Well, even if you are going to write your code in a way that supports batch sizes, you're probably using this so that you can calculate gradients using the `jax.grad` function. You can only take a gradients of functions that output a scalar value. So this is the only way you can vectorize the computation without doing explicit loops.
