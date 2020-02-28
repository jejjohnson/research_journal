```python
def kernel_product(x,y, mode = "gaussian", s = 1.):
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    xmy = ((x_i-y_j)**2).sum(2)

    if   mode == "gaussian" : K = torch.exp( - xmy/s**2) )
    elif mode == "laplace"  : K = torch.exp( - torch.sqrt(xmy + (s**2)))
    elif mode == "energy"   : K = torch.pow(   xmy + (s**2), -.25 )

    return torch.t(K)

```