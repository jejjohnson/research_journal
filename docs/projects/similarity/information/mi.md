# Mutual Information

> How much information one random variable says about another random variable.

- [Intiution](#intiution)
- [Full Definition](#full-definition)
- [Code](#code)
- [Supplementary](#supplementary)
  - [Information](#information)
    - [Intuition](#intuition)
- [Supplementary](#supplementary-1)


---

## Intiution

* Measure of the amount of information that one RV contains about another RV
* Reduction in the uncertainty of one rv due to knowledge of another
* The intersection of information in X with information in Y

---

## Full Definition

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

**Sources**:
* [Scholarpedia](http://www.scholarpedia.org/article/Mutual_information)

---

## Code

1. We need a PDF estimation...


2. Normalize counts to probability values

```python
pxy = bin_counts / float(np.sum(bin_counts))
```

3. Get the marginal distributions

```python
px = np.sum(pxy, axis=1) # marginal for x over y
py = np.sum(pxy, axis=0) # marginal for y over x
```

4. Joint Probability

---

## Supplementary

### Information


#### Intuition

> Things that don't normally happen, happen.




---



---

## Supplementary

* [MI w. Numpy](https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy)
* [Predictions and Correlations in Complex Data](https://www.freecodecamp.org/news/how-machines-make-predictions-finding-correlations-in-complex-data-dfd9f0d87889/)