# Anomaly Detection

```python
# define threshold
instances = 4 # 4%

# calculate densityes
densities = model.score_samples(X)

# calculate density threshold
density_threshold = np.percentile(densities, instances)

# reproduce the anomalies
anomalies = X[densities < densities_threshold]
```

For more examples, see the [pyOD]() documentation. In particular:

* [`predict_proba`](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/base.html#BaseDetector.predict_proba) - predict the probability of a sample being an outlier.
* [`predict`](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/base.html#BaseDetector.predict) - predict if a sample is an outlier or not.