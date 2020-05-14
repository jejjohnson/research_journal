# PyTorch Lightning

### Debugging Tips

* There are 5 validation runs before the training loop starts (built-in)
* `fast_dev_run` - runs 1 batch of training and testing data (like compiling)
* `overfit_pct=0.01` - can my model overfit on 1% of my data? The loss should go to 0...
  * `train_percent_check=0.1, val_percent_check=0.01` - same thing but with specific numbers


### Running Accuracy


First in the `init` portion of your code:

```python
self.last_accuracies = []
```

Then in the validation or train step:

```python
# append accuracies
self.last_accuracies.append(...)
val_acc = ...
```

Get all the accuracies:

```python
# mean of n previous steps
torch.stack(self.last_accuracies[-n_previous_steps:].mean())
```

### Custom Callbacks

You can use the built-in method:

```python
trainer = Trainer(early_stop_callback=True)
```

This method will look for the 'val_loss' that can be found within the training.

Alternatively, you can use a custom callback where you can set whatever loss you would like:

```python
early_stop_callback = EarlyStopping(
  monitor='tc_loss',
  min_delta=0.0,
  patience=3,
  verbose=False,
  mode='min'
)

trainer = Trainer(early_stop_callback=early_stop_callback)
```


### Tensorboard