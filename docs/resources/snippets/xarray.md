# XArray

## Seasonal Grouping

### Extract Time Series (from Location)

```python
time_series = data.isel(x=1000, y=1000).to_pandas().dropna()
```

### Mean Across Multiple Dimensions

```python
data.mean(dim=['lat', 'lon'])
```