# Sweeps using Weights and Biases

In this tutorial, we're going to do a simple script which will allow us to do sweeps using weights and biases. 



### Import Libraries

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
```


### ML Model


```python
 def __init__(self):
   super(LightningMNISTClassifier, self).__init__()

   # mnist images are (1, 28, 28) (channels, width, height)
   self.layer_1 = torch.nn.Linear(28 * 28, 128)
   self.layer_2 = torch.nn.Linear(128, 256)
   self.layer_3 = torch.nn.Linear(256, 10)
 def prepare_data(self):
   # prepare transforms standard to MNIST
   MNIST(os.getcwd(), train=True, download=True)
   MNIST(os.getcwd(), train=False, download=True)

 def train_dataloader(self):
   #Load the dataset
   mnist_train = DataLoader(self.mnist_train, batch_size=64)
   return mnist_train

 def val_dataloader(self):
   #Load val dataset
   mnist_val = DataLoader(self.mnist_val, batch_size=64)
   return mnist_val

 def test_dataloader(self):
   #Load test data
   mnist_test = DataLoader(mnist_test, batch_size=64)
   return mnist_test
```