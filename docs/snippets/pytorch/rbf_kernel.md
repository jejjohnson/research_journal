
This snippet showcases using PyTorch and calculating a kernel function. Below I have a sample script to do an RBF function along with the gradients in PyTorch.


```python
from typing import Union
import numpy as np
import torch            # GPU + autodiff library
from torch.autograd import grad

class RBF:
    def __init__(
        self, 
        length_scale: float=1.0, 
        signal_variance: float=1.0, 
        device: Union[Bool,str]=None) 
    -> None:
        
        # initialize parameters
        self.length_scale = torch.tensor(
            length_scale, dtype=torch.float32, device=self.device,
            requires_grad=True
        )
        self.signal_variance = torch.tensor(
            signal_variance, dtype=torch.float32, device=self.device,
            requires_grad=True
        )
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        
    def __call__(
            self, 
            X: np.ndarray, 
            Y: Union[Bool, np.ndarray]=None) 
        -> np.ndarray:
        
        # convert inputs to pytorch tensors
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if Y is None:
            Y = X
        else:
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
            
        # Divide by length scale
        X = torch.div(X, self.length_scale)
        Y = torch.div(Y, self.length_scale)
        
        # Re-indexing
        X_i = X[:, None, :] # shape (N, D) -> (N, 1, D)
        Y_j = Y[None, :, :] # shape (N, D) -> (1, N, D)
        
        # Actual Computations
        sqd     = torch.sum( (X_i - Y_j)**2, 2)         # |X_i - Y_j|^2
        K_qq    = torch.exp( -0.5 * sqd )               # Gaussian Kernel
        K_qq    = torch.mul(self.signal_variance, K_qq) # Signal Variance
        
        return K_qq.detach().to_numpy()
    
    def gradient_X(self, X):
    
        return None
    
    def gradient_X2(self, X):
        
        return None
    
    def gradient_XX(
            self, 
            X: np.ndarray, 
            Y: Union[Bool, np.ndarray]=None) 
        -> np.ndarray:
        
        # Convert to tensor that requires Grad
        X = torch.tensor(
            length_scale, dtype=torch.float32, device=self.device,
            requires_grad=True
        )
        
        if Y is None:
            Y = X
        else:
            Y = torch.tensor(
                Y, dtype=torch.float32, device=self.device,
                requires_grad=True
            )
        # compute the gradient kernel w.r.t. to the two inputs
        J = grad(self.__call__(X, Y))
    
        return J
        
    def gradient_XX2(self, X, Y=None):
        
        return None
```


Below we can see how one would actually do that in practice.


```python
# With PyTorch, using the GPU is simple
use_gpy     = torch.cuda.is_available()
dtype       = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

N = 5000    # cloud of 5,000 points
D = 3       # 3D

q = np.random.rand(N, D)
p = np.random.rand(N, D)
s = 1.

# Store arbitrary arrays on the CPU or GPU:
q = torch.from_numpy(q).type(dtype)
p = torch.from_numpy(p).type(dtype)
s = torch.Tensor([1.]).type(dtype)

# Tell PyTorch to track the variabls "q" and "p"
q.requires_grad = True
p.requires_grad = True

# Rescale with length_scale
q = torch.div(q, s)

# Re-indexing
q_i = q[:, None, :] # shape (N, D) -> (N, 1, D)
q_j = q[None, :, :] # shape (N, D) -> (1, N, D)

# Actual Computations
sqd     = torch.sum( (q_i - q_j)**2, 2) # |q_i - q_j|^2
K_qq    = torch.exp( -sqd / s**2 )      # Gaussian Kernel
v       = K_qq @ p                      # matrix mult. (N,N) @ (N,D) = (N,D)

# Automatic Differentiation 
[dq, dp] = grad( H, [q,p] )

# Hamiltonian H(q,p): .5*<p,v>
H   = .5 * torch.dot( p.view(-1), v.view(-1) )
```


**Source**: Presentation for [Autograd](http://www.math.ens.fr/~feydy/Talks/autodiff_appliedmaths/AutoDiff_AppliedMaths.pdf) and mathematics.
