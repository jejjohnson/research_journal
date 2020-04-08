# Histograms in PyTorch


It's not very well known but there is a histogram function in PyTorch.


**Function**

```python
# histogram parameters
bins = 4
bounds = (4, 0)

# calculate the histogram
hist = torch.histc(torch.tensor([1., 2., 1.]), bins=bins, min=bounds[0], max=bounds[1])

# normalize histogram to sum to 1
hist = hist.div(hist.sum())
```

**Calculating Bin Edges**

Unfortunately, we have to do this manually as the pytorch function doesn't spit out the entire function.

```python
# calculate the bin edges
bin_edges = torch.linspace(bounds[0], bounds[1], steps=bins)
```