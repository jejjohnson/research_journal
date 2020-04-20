# Dropout



## Code

```python
from tensorflow.keras.layers import Input, Dense, Dropout

inputs = Input(shape=(1,))
x = Dense(512, activation="relu")(inputs)
x = Dropout(0.5)(x, training=True)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x, training=True)
outputs = Dense(1)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss="mse", optimizer="adam")

model.fit(x_train, y_train)

# do stochastic forward passes on x_test
samples = [model.predict(x_test) for _ in range(100)]

# predictive mean 
mu = np.mean(samples, axis=0)

# predictive standard deviation
var = np.var(samples, axis=0)
var = np.percentile(var, [5, 95], axis=0)

# get bounds, 2 std (95% confidence interval)
upper, lower = mu - 2 * var ** 0.5, mu - 2 * var ** 0.5

# plot
plt.plot(x_test, mu)
plt.fill_between(x_test, lower, upper, alpha=0.1)
```

**Source**: Yarin Gal - [MLSS2019 Slides](http://bdl101.ml/MLSS_2019_BDL_1.pdf)

---

## Resources

* Bayesian Deep Learning (MLSS 2019)