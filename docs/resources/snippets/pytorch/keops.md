# KeOps - Gaussian Kernel 

```python

import torch
from pykeops.torch import Kernel, kernel_product

# Geneerate the data as pytorch tensors
x = torch.randn(1000, 3, requires_grad=True)
y = torch.randn(2000, 3, requires_grad=True)
b = torch.randn(2000, 2, requires_grad=True)

# ARD length_scale
length_scale = torch.tensor([.5], requires_grad=True)
params = {
    "id"    : Kernel("gaussian(x,y)"),
    "gamma" : 1./length_scale**2,
}

# Differentiable wrt x, y, b, length_scale
a = kernel_product(params, x, y, b)
```