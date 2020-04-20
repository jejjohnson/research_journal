# Jit


## Fixing some Arguments


So `jax.jit` doesn't play nice sometimes when you try to compile functions with arguments that are functions themselves. A lot of times you'll get an error. One trick I learned to get around this is to "partially fit" the function. Let's do an example:

```python
def predict(func, params, input_vector):
    # intermediate function
    x_trans = func(params, input_vector)

    # real function
    output = params['w'] * input_vector + x_trans

    return output
```

If we try to jit compile this...well it won't less us and will give us an error message.

```python
predict_f_jitted = jax.jit(predict)
```

Instead we need to make some arguments static so that once the function is compiled, it doesn't change those particular arguments. In our case, it would be the func. So to do this, we can do the following:

```python
predict_f_jited = jax.jit(predict, static_argnums=(0,))
```

An now, we won't get an error message! Now, those functional experts will probably say that this isn't a good way to do function programming and we should be using context. For example:

```python
def func(params, input_vector):
    # do stuff
    ...
    return output

def predict(params, input_vector):
    # intermediate function
    x_trans = func(params, input_vector)

    # real function
    output = params['w'] * input_vector + x_trans

    return output
```

In this case, we should be able to apply the `jit` function because we have saved the previous function via the context. I personally don't like this but mainly because I'm not used to it. I like my functions to be relatively independent and I'm not very good at managing context. I imagine it's super useful when you want to make a script that has all of the functions that you need all within the `.py` file. That way you can handle the context as you progress through the script. For if you want independent files (which I like for sanity purposes), I don't see how this is possible. But again, it's a personal preference. Plus, I'm not an expert by any means.

### Using the decorator

We can also use the `functools.partial` decorator to wrap our function so that once it's called, it will automatically compile the function.

```python
from functools import partial

@partial(jax.jit, static_argnums=(0))
def predict(func, params, input_vector):
    # intermediate function
    x_trans = func(params, input_vector)

    # real function
    output = params['w'] * input_vector + x_trans

    return output
```

I use this quite often if I have functions that I know I won't change certain parameters the moment I start using them. It also goes well with my style of having a lot of individual functions in separate files.