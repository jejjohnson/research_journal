# PyTorch Ideas


## 1. Neural Networks from scratch


* [BNN](https://matthewmcateer.me/blog/a-quick-intro-to-bayesian-neural-networks/)


## 2.


---

## PyTorch Lightning

### Debugging Tips

* There are 5 validation runs before the training loop starts (built-in)
* `fast_dev_run` - runs 1 batch of training and testing data (like compiling)
* `overfit_pct=0.01` - can my model overfit on 1% of my data? The loss should go to 0...
  * `train_percent_check=0.1, val_percent_check=0.01` - same thing but with specific numbers