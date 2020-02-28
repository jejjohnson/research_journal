# Loops with TQDM

A simple way to use a nice progress bar instead of polluting your screen with print statements.

```python
import tqdm

def train(xtrain, ytrain, model, criterion, optimizer, n_epochs = 1_000):

    with tqdm.trange(n_epochs) as bar:
        for epoch in bar:  # loop over the dataset multiple times
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(xtrain)
            loss = criterion(outputs, ytrain)
            loss.backward()
            optimizer.step()
    
            # print statistics
            postfix = dict(Loss=f"{loss.item():.3f}")
            bar.set_postfix(postfix)
```

**Source**: DeepBayes.ru 2019 [Notebook](https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/GP/gp_solution.ipynb)
