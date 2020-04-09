# Logistic Distribution



## Cheat Sheet


**PDF Logistic Distribution**

$$f(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \frac{\exp(-z)}{\sigma(1 + \exp(-z))^2}$$

where $z = \frac{(x-\mu)}{\sigma}$.

* Support: $(-\infty, \infty)$

---

**CDF Logistic Distribution**

$$F(x) = \frac{1}{1 + \exp(-x)} = \frac{1}{1 + \exp(-z)}$$

where $z = \frac{(x-\mu)}{\sigma}$.

* Support: $(-\infty, \infty) \rightarrow [0, 1]$

---

**Quantile Function Logistic Distribution**

$$F^{-1}(x) = \log\left(\frac{p}{1-p}\right) = \mu + \sigma_{\log} \log \left( \frac{p}{1-p} \right)$$

where $p \sim \mathcal{U}([0,1])$.

Inverse sampling


---

**Log SoftMax**

$$\text{LogSoftmax}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_i)} \right)$$

PyTorch Function - `Functional.log_softmax`

---

**Sigmoid**

$$\text{Sigmoid}(x) = \frac{1}{1 +\exp(-x)} $$

PyTorch Function - `Functional.sigmoid`

---

**Log Sigmoid**

$$\text{LogSigmoid}(x) = \log \left( \frac{1}{1 +\exp(-x)} \right)$$

PyTorch Function - `Functional.logsigmoid`

---

**Log Sum Exponential**

$$\text{LogSumExp}(x)_i = \log \sum_j \exp(x_{ij})$$

PyTorch Function - `torch.logsumexp`

---

**SoftPlus**

$$\text{SoftPlus}(x) = \frac{1}{\beta}\log \left(1 + \exp(\beta x) \right)$$

* PyTorch Function - `Function.softplus`

---