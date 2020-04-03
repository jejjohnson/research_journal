# DL 4 Researchers Part I - NN from scratch

Author: J. Emmanuel Johnson
Email: jemanjohnson34@gmail.com
PyTorch Colab Notebook: [Link](https://colab.research.google.com/drive/1GiQ8Y6awgKQ-n1h6lMsqETSUqpYDVn0p)

---

Sources:
* [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) - PyTorch

---

- [DL 4 Researchers Part I - NN from scratch](#dl-4-researchers-part-i---nn-from-scratch)
  - [Import Data](#import-data)
    - [Convert Inputs to Tensors](#convert-inputs-to-tensors)
      - [** PyTorch **](#pytorch)
      - [** TensorFlow **](#tensorflow)
  - [Parameters](#parameters)
      - [** PyTorch **](#pytorch-1)
    - [Tracking Gradients](#tracking-gradients)
      - [** PyTorch **](#pytorch-2)
  - [Model](#model)
      - [** PyTorch **](#pytorch-3)
    - [Loss Function](#loss-function)
      - [** PyTorch **](#pytorch-4)
  - [Training](#training)
      - [** PyTorch **](#pytorch-5)
  - [Appendix](#appendix)
      - [Extra Python Packages Used](#extra-python-packages-used)

---

## Import Data

We will be using the california housing dataset. We will do some standard preprocessing including Normalizing the inputs, removing the mean from the outputs, and splitting the data into train, validation and testing.

```python
# Import Data
cal_housing = datasets.fetch_california_housing()

# extract training and targets
X = cal_housing.data
y = cal_housing.target

# Normalize Input data
X = StandardScaler().fit_transform(X)

# Remove mean from Output data
y = StandardScaler(with_std=False).fit_transform(X)

# split data into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.1, random_state=0)

# split training data into train and validation
Xtrain, Xvalid, ytrain, yvalid = train_test_split(
    Xtrain, ytrain, train_size=0.8, random_state=123
)
```

### Convert Inputs to Tensors

We need to convert the data from `np.ndarray` to a `Tensor`.

<!-- <details> -->
<!-- tabs:start -->

#### ** PyTorch **

```python
# Create a torch tensor from the data
x_train, y_train, x_valid, y_valid = map(
    torch.FloatTensor, (Xtrain, ytrain, Xvalid, yvalid)
)
```

#### ** TensorFlow **

```python
# Create a torch tensor from the data
x_train, y_train, x_valid, y_valid = map(
    tf.convert_to_tensor, (Xtrain, ytrain, Xvalid, yvalid),
)
```
<!-- tabs:end -->

<!-- </details> -->


!> One small thing I've noticed is that TensorFlow handles numpy arrays much better than PyTorch. PyTorch is very strict: you need to convert your data to a PyTorch tensor before you start using PyTorch functions. TensorFlow is a bit more flexible sometimes and can convert your data into tf.tensors.

---

## Parameters

The first thing we need to do is define the components that we're working with. We are doing a simple linear regression problem so we only need the following components:

* weight matrix: $\mathbf{W} \in \mathcal{R}^{D \times 1}$ 
* bias vector: $b \in \mathcal{R}^{D \times 1}$

<!-- tabs:start -->

#### ** PyTorch **

```python
import math

# dimensions for parameters
input_dim = 8
output_dim = 1
n_samples = x_train.shape[0]

# weight 'matrix'
weights = torch.randn(input_dim, output_dim)  / math.sqrt(input_dim)

# bias vector
bias = torch.zeros(output_dim)
```

<!-- tabs:end -->

### Tracking Gradients

<!-- tabs:start -->

#### ** PyTorch **

```python
weights.requires_grad_()
bias.requires_grad_()
```

<!-- tabs:end -->


---

## Model 

So again, the model is simple:

$$y = \mathbf{x}_b \mathbf{W} + b$$

<!-- tabs:start -->

#### ** PyTorch **

```python
# define the model as a function
def model(x_batch: torch.tensor):
    return x_batch @ weights + bias

batch_size = 64

# mini-batch from training data
xb = x_train[:batch_size] 

# predictions
preds = model(xb)

# check if there is grad function
preds[0]
```

You should get the following output:

```
tensor([0.3591], grad_fn=<SelectBackward>)
```

That `grad_fn` lets you know that we're tracking the gradients.

<!-- tabs:end -->


### Loss Function 

We're doing a simple loss: mean squared error.

$$\mathcal{L}_{mse} = \frac{1}{N}\sum_{i=1}^{N}\left( \hat{y}_i - y_i \right)^2$$

<!-- tabs:start -->

#### ** PyTorch **

```python
# define mse loss
def mse_loss(input: torch.tensor, target: torch.tensor):
    return torch.mean((input - target) ** 2)

# set loss function to mse
loss_func = mse_loss

# get sample batch
yb = y_train[:batch_size]

# get loss
loss = loss_func(preds, yb)

# check if there is grad function
print(loss)
```

<!-- tabs:end -->


## Training

<!-- tabs:start -->

#### ** PyTorch **

```python
batch_size = 100
learning_rate = 0.01
epochs = 20
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
            ypred = model(xbatch)

            # loss
            loss = loss_func(ypred, ybatch)

            # Loss back propagation
            loss.backward()

            # add running loss
            losses.append(loss.item())

            # manually calculate gradients
            with torch.no_grad():

                # update the weights individually
                weights -= learning_rate * weights.grad
                bias    -= learning_rate * bias.grad

                # zero the weights, bias parameters
                weights.grad.zero_()
                bias.grad.zero_()

            # Update status bar
            postfix = dict(
                Epoch=f"{iepoch}", 
                Loss=f"{loss.item():.3f}",
                )
            bar.set_postfix(postfix)
```

Your output will look something like this.

```
100%|██████████| 20/20 [00:08<00:00,  2.50it/s, Epoch=19, Loss=0.545]
```

<!-- tabs:end -->

---

## Appendix


#### Extra Python Packages Used

**[scikit-learn]()**

The default machine learning package for python. It has a LOT of good algorithms and preprocessing features that are useful. Apparently the biggest use of the sklearn library is the `train_test_split` function. I'm definitely guilty of that too.

**[tqdm]()**

A  nice full featured status bar which can help eliminate some of the redundant outputs.

**[typing]()**

A built-in type checker. This allows one to type check your inputs. It helps with code readability and to catch silly errors like having bad inputs.