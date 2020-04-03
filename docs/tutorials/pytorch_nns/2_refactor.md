# DL 4 Researchers Part II - Refactoring


## Model (and Parameters)

**Old Way**

<details>

<!-- tabs:start -->

#### ** PyTorch **

```python
from torch import nn
import math

# dimensions for parameters
input_dim = 8
output_dim = 1
n_samples = x_train.shape[0]

# weight 'matrix'
weights = nn.Parameter(
    torch.randn(input_dim, output_dim) / math.sqrt(input_dim),
    requires_grad=True
) 

# bias vector
bias = nn.Parameter(
    torch.zeros(output_dim),
    requires_grad=True
)

# define model
def model(x_batch: torch.tensor):
    return x_batch @ weights + bias

# set linear model
lr_model = model
```

<!-- tabs:end -->

</details>

<!-- tabs:start -->

#### ** PyTorch **

```python
class LinearModel(nn.Module):
    """A Linear Model

    Parameters
    ----------
    input_dim : int,
        The input dimension for the linear model
        (# input features)
    
    output_dim : int,
        the output Dimension for the linear model
        (# outputs)

    Attributes
    ----------
    weights : torch.Tensor (input_dim x output_dim)
        the parameter for the linear model weights

    bias : torch.Tensor (output_dim)
        the parameter for the linear model bias
    
    Methods
    -------
    forward : torch.tensor (input_dim x output_dim)
        the forward pass through the linear model
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # weight 'matrix'
        self.weights = nn.Parameter(
            torch.randn(input_dim, output_dim) / math.sqrt(input_dim),
            requires_grad=True
        ) 

        # bias vector
        self.bias = nn.Parameter(
            torch.zeros(output_dim),
            requires_grad=True
        )

    def forward(self, x_batch: torch.tensor):
        return x_batch @ self.weights + self.bias

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

lr_model = LinearModel(input_dim, output_dim)
```
So we have effectively encapsulated that entire parameter definition within a single understandable function. We have the parameters defined when we initialize the `model` and then we have the `forward` method which allows us to perform the operation.

<!-- tabs:end -->

---

## Loss Function

We can also look and use the built-in loss functions. The `mse` is a very common loss function so it should be available within the library.

#### Old Way


<!-- tabs:start -->

#### ** PyTorch **

In PyTorch, we need to look at the `nn.functional.mse_loss` module or the `nn.MSELoss()`. The latter has more options as it is a class and not a function but the former will do for now. So we can change the old way:

<details>


```python
def mse_loss(input: torch.tensor, target: torch.tensor):
    return torch.mean((input - target) ** 2)
```
</details>

to a simplified version.


```python
import torch.nn.functional as F

# set loss function to mse
loss_func = F.mse_loss
```

<!-- tabs:end -->

---

## Optimizer

Another refactor opportunity is to use a built-in optimizer. I don't want to have to calculate the gradient for each of the weights multiplied by the learning rate.

<!-- tabs:start -->

#### ** PyTorch **

```python
from torch import optim

learning_rate = 0.01

# use stochastic gradient descent
opt = optim.SGD(lr_model.parameters(), lr=learning_rate)
```

<!-- tabs:end -->


---

## Training

So after all of that hard work, the training procedure will look a lot cleaner because we have encapsulated a lot of operations using the built-in operations. Now we can focus on other things.

#### Old Way 


#### New Way


<!-- tabs:start -->

#### ** PyTorch **

```python
batch_size = 100
epochs = 10
n_samples = x_train.shape[0]
n_batches = (n_samples - 1) // batch_size + 1
losses = []


with tqdm.trange(epochs) as bar:
    # Loop through epochs with tqdm bar
    for iepoch in bar:
        # Loop through batches
        for idx in range(n_batches):

        # get indices for batches
            start_idx = idx * batch_size
            end_idx   = start_idx + batch_size

            xbatch = x_train[start_idx:end_idx]
            ybatch = y_train[start_idx:end_idx]

            # predictions
            ypred = lr_model(xbatch)

            # loss
            loss = loss_func(ypred, ybatch)

            # add running loss
            losses.append(loss.item())

            # Loss back propagation
            loss.backward()

            # optimize weights
            opt.step()
            opt.zero_grad()
            

            postfix = dict(
                Epoch=f"{iepoch+1}", 
                Loss=f"{loss.item():.3f}",
                )
            bar.set_postfix(postfix)
```

<!-- tabs:end -->

---

##  Datasets and DataLoaders

Now there are some extra things we can do to reduce the amount of code and make this neater. We can use `Datasets` and `DataLoaders`. 



<!-- tabs:start -->

#### ** PyTorch **

### Dataset

In PyTorch, the `Dataset` helps us to do index and slice through our data. It also can combine inputs and outputs so that we only have to slice through a single dataset. It can even convert your `np.ndarray` dataset to a Tensor automatically. So instead of 

```python
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

lr_model, opt = get_lr_model()

batch_size = 100
epochs = 10
n_samples = x_train.shape[0]
n_batches = (n_samples - 1) // batch_size + 1
losses = []


with tqdm.trange(epochs) as bar:
    # Loop through epochs with tqdm bar
    for iepoch in bar:
        # Loop through batches
        for idx in range(n_batches):

        # get indices for batches
            
            start_idx = idx * batch_size
            end_idx   = start_idx + batch_size

            # Use Dataset to store training data
            xbatch, ybatch = train_ds[start_idx:end_idx]

            # predictions
            ypred = lr_model(xbatch)

            # loss
            loss = loss_func(ypred, ybatch)

            # add running loss
            losses.append(loss.item())

            # Loss back propagation
            loss.backward()

            # optimize weights
            opt.step()
            opt.zero_grad()
            

            postfix = dict(
                Epoch=f"{iepoch+1}", 
                Loss=f"{loss.item():.3f}",
                )
            bar.set_postfix(postfix)
```

### DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

batch_size = 100

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)

# initialize model
lr_model, opt = get_lr_model()

epochs = 10
losses = []

with tqdm.trange(epochs) as bar:
    # Loop through epochs with tqdm bar
    for iepoch in bar:
        # Loop through batches
        for xbatch, ybatch in train_dl:

            # predictions
            ypred = lr_model(xbatch)

            # loss
            loss = loss_func(ypred, ybatch)

            # add running loss
            losses.append(loss.item())

            # Loss back propagation
            loss.backward()

            # optimize weights
            opt.step()
            opt.zero_grad()
            

            postfix = dict(
                Epoch=f"{iepoch+1}", 
                Loss=f"{loss.item():.3f}",
                )
            bar.set_postfix(postfix)
```

### Validation set

So because it's so easy, we can now add that validation set. I would have dreaded doing that before due to the lengthy code. But now, it's a piece of cake.

```python
from torch.utils.data import TensorDataset, DataLoader

# training set
batch_size = 100
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# validation set
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(train_ds, batch_size=batch_size)

# initialize model
lr_model, opt = get_lr_model()

epochs = 10
train_losses, valid_losses = [], []

with tqdm.trange(epochs) as bar:
    # Loop through epochs with tqdm bar
    for iepoch in bar:
        
        # put in training mode
        lr_model.train()

        # Loop through batches
        for xbatch, ybatch in train_dl:

            # predictions
            ypred = lr_model(xbatch)

            # loss
            loss = loss_func(ypred, ybatch)

            # add running loss
            train_losses.append(loss.item())

            # Loss back propagation
            loss.backward()

            # optimize weights
            opt.step()
            opt.zero_grad()
            

            postfix = dict(
                Epoch=f"{iepoch+1}", 
                Loss=f"{loss.item():.3f}",
                )
            bar.set_postfix(postfix)

        # put in evaluation model
        lr_model.eval()
        with torch.no_grad():
            for xbatch, ybatch in valid_dl:
                loss = loss_func(lr_model(xbatch), ybatch)

                valid_losses.append(loss) 
```

<!-- tabs:end -->

## Appendix

#### Python Concepts

**Python Classes**

More information can be found [here]().

**Comments**

Do them. Always. It might seem like a sunk cost, but it will save you time in the end. 

