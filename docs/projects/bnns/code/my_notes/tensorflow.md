# TensorFlow 2.0

Author: J. Emmanuel Johnson
Email: jemanjohnson34@gmail.com

These are notes that I took based off of lectures 1 and 2 given by Francois Chollet.

---

## Architecture



#### 1. Engine Module

This is basically the model definition. It has the following parts

* `Layer`
* `Network` - this contains the DAG of Layers (internal component)
* `Model` - this contains the network and is used to do the training and evaluation loops
* `Sequential` - wraps a list of layers

#### 2 Various Classes (and subclasses)

* Layers
* Metric
* Loss
* Callback
* Optimizer 
* Regularizers, Constraints?

---

## `Layer` Class



This is the core abstraction in the API. Everything is a `Layer` or it at least interacts closely with the `Layer`.

#### What can it do?

**Computation**

This manages the computation. It takes in batch inputs / batch outputs. 

* Assumes no interactions between samples
* Eager or Graph execution
* `Training` and `Inference` model
* Masking (e.g. time series, missing features)

**Manages State**

This keeps track of what's trainable or not trainable.

```python
class Linear(tf.keras.Layer):
  def __init__(self):
    super().__init__()
    self.weights = ...trainable
    self.bias = ...not trainable
```



**Track Losses & Metrics**

Up

```python
class Linear(tf.keras.Layer):
  def call(self, x):
    # calculate kl divergence
    kl_loss = ...
    # add loss
    self.add_loss(...)
```



* Type Checking
* Frozen or UnFrozen (fine-tuning, batch-norm, GANS)
* Can build DAGs - Sequential Form
* Mixed Precio



### What do they not do?

* Gradients
* Device Placement
* Distribution-specific logic
* Only batch-wise computation.

### Basic Layer

We are going to create a base layer

```python
# create linear layer
class Linear(tf.keras.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        
        # weights variable
        w_init = tf.random_normal_initializer()(shape=(input_dim, units))
        self.w = tf.Variable(
            initial_value=w_init,
            trainable=True
       	)
        
        # bias parameter
        b_init = tf.zeros_initializer()(shape=(units,))
        self.b = tf.Variable(
        	initial_value=b_init,
        	trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + b
    
# data
x_train = tf.ones(2, 2)

# initialize linear layer
linear_layer = Linear(4, input_dim=2)

# same thing as linear_layer.call(x)
y = linear_layer(x)
```



#### Better Basic Layer

```python
class Linear(tf.keras.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__()
        self.units = units
```

Notice how we didn't construct the weights when we initialized the class (constructor). This is nice because now we can construct our layer without having to know what the input dimension will be. We can simply specify the units. Instead we create a `build` method and that has the weights specified.

```python
    def build(self, input_shape):
        # Weights variable
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), 
            initializer='random_normal',
            trainable=True
        )
        # Bias variable
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

The rest doesn't change. We can initialize the liner layer just with the units. This is called 'Lazy loading'

```python
linear_layer = Linear(32)
```

It will call `.build(x.shape)` to get the dimensions of the dataset.

```python
y = linear_layer(x)
```

### Nested Layers

We can nest `Layers` (as many) layers as we want actually. For example:

#### Multi-Layer Perceptron

```python
class MLP(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.linear = Linear(units)

    def call(self, inputs):
        x = self.linear(inputs)
        return x
```

#### MLP Block

```python
class MLPB(Layer):
    def __init__(self):
        super().__init__()
        self.mlp_1 = MLPBlock(32)
        self.mlp_2 = MLPBlock(32)
        self.mlp_3 = MLPBlock(1)
    
    def call(self, inputs):
        x = self.mlp_1(x)
        x = self.mlp_2(x)
        return x = self.mlp_3(x)
```

---





### Basic Training

So assuming that we have our linear layer, we can do some basic training procedure.

```python
# initialize model
lr_model = Linear(32)
# loss function
loss_fn = tf.keras.losses.MSELoss()
# optimizer
optimizer = tf.keras.optimizers.Adam()
# Loop through dataset
for x, y in dataset:
    with tf.GradientTape() as tape:
        # predictions for minibatch
        preds = linear_model(x)
        # loss value for minibatch
        loss = loss_fn(y, preds)
    # find gradients
    grads = tape.gradients(loss, lr_model.trainable_weights)
    # apply optimization
    optimizer.apply_gradients(zip(grads, lr_model.trainable_weights))
```



---

### Losses

We can add losses on the fly. For example, we can add a small activation regularizer in the call function for the MLP layer that we made above:

```python
class MLP(Layer):
    def __init__(self, units=32, reg=1e-3):
        super().__init__()
        self.linear = Linear(units)
        self.reg = reg
    def call(self, inputs):
        x = self.linear(inputs)
        x = tf.nn.relu(x)
        self.add_loss(tf.reduce_sum(output ** 2) * self.reg)
        return x
```

Now when we call the layer, we get the activation loss.

```python
mlp_layer = MLP(32)
y = mlp_layer(x)
```

Now it gets reset everytime we call it.

#### Modified Training Loop

```python
mlp_model = MLP(32)                    	# initialize model
loss_fn = tf.keras.losses.MSELoss()    	# loss function
optimizer = tf.keras.optimizers.Adam() 	# optimizer
# Loop through dataset
for x, y in dataset:
    with tf.GradientTape() as tape:
        preds = mlp_model(x)          	# predictions for minibatch
        loss = loss_fn(y, preds) 		# loss value for minibatch
        loss += sum(mlp_model.losses)	# extra losses from forward pass
    # find gradients
    grads = tape.gradients(loss, mlp_model.trainable_weights)
    # apply optimization
    optimizer.apply_gradients(zip(grads, mlp_model.trainable_weights))
```



Useful for:

* KL-Divergence
* Weight Regularization
* Activation Regularization

**Note**: There is some context. The inner layers are also reset when their parent layer is called.

---

### Serialization

```python
class Linear(tf.keras.Layer):
    def __init__()
	...
    
	def get_config(self):
        config  super().get_config()
        config.update({'units': self.units})
        return config
```



---

### Training Mode

Allows you to do training versus inference mode. You simply need to add an extra argument in the `cal()` method.

```python
...
def call(self, x, training=True):
    if training:
        # do training stuff
    else:
        # do inference stuff
    return x
```





Some good examples:

* Batch Normalization
* Probabilistic Models (MC Variational Inference)

---

## `Model` Class

This handles top-level functionality. The `Model` class does everything the `Layer` class can do, i.e. it is the same except with more available methods. In the literature, we refer to this as a "model", e.g. a deep learning model, a machine learning model, or as a "network", e.g. a deep neural network.

In the literature, we refer to a `Layer` as something with a closed sequence of operations. For example a convolutional layer or a recurrent layer. Sometimes we also refer layers within layers as a block. For example a ResNet block or an Attention block. 

So ultimately, you would define the `Layer` class to do the inner computation blocks and the `Model` class to do the outer model with what you do to train and save.

**Training functionality**

* `.compile()`
* `.fit()`
* `.evaulate()`
* `.predict()`

**Saving**

We have the `.save()` method which includes:

* configuration (topology)
* state (weights)
* optimiser

**Summarization & Visualization**

* `.summary()`
* `plot_model()`

### Compile

This option give configurations:

* optimizer
* Loss



When you have the model class and you run `.compile()`, you are running the graph in graph execution model. So you are basically compiling the graph. If we want to run it eagerly: we need to set the paramter `run_eagerly` to be `True`.



```python
mlp = MLP()
mlp.compile(optimizer=Adam(), loss=MSELoss(),run_eagerly=True)
```



### Fit

How the data will be fit: The training procedure.

* Callbacks
* Data
* Epochs

---

## Functional Model





