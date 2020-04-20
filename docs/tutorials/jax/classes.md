# Classes

There was an interesting example where someone had asked about how one can deal with classes. I have been a supporter of classes for a while now but now I've started to rethinking my convictions. OOP code is quite difficult to maintain by yourself and if you don't do the abstract correctly, you run into problems later on. So...I've started to branch out a bit. Let's get into it.

Let's say we have two parameters that we want to update. So naturally, I would build a class:

```python
class Idea:
    def __init__(self, w, b):
        self.w = w
        self.b = b
```

Now we want to do something. Let's say it's a neural network and we want to perform a transformation. So let's implement a predict function.

```python
class NN:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def predict(self, input):
        return np.dot(self.w, x) + self.b
```

Easy enough, now we would like to train this. So we need a step function which handles the gradients and another function for the training loop.

```python
class NN:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def predict(self, input):
        return np.dot(self.w, x) + self.b
    def step(self, input):

```


---

## Functional Version

```python
Params = namedtuple("Params", ["w", "b"])


def predict(params, inputs):
    return np.dot(params.w, x) + params.b

def step(params, inputs, lr=0.1):
    # get gradients
    grads = dloss(params, inputs)

    # update parameters
    w_new = params.w - lr * grad.w
    b_new = params.b - lr * grad.b

    # return new parameter tuple
    return Params(w_new, b_new)

for i in range(1_000):
    params = step(params, inputs)
```


## Resources

* [Github Issue](https://github.com/google/jax/issues/1567)
  > An example straight from the developers of how to create classes using jax. They're not full traditional classes but they do it in such a way that it works.


