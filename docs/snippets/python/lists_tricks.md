# Tricks with Lists



## Unpacking a List of tuples

I've found this useful in situations where I need to do parallel processing and my output is a tuple of 2 elements. So for example, let's have the following.
```python
list_of_tuples = ('1', 'a'), ('2', 'b'), ('3', 'c'), ('4', 'd')
```

Now I would like to unpack this into 2 lists. I can do this like so:

```python
list1, list2 = zip(*list_of_tuples)
```

So if I print the lists, I will get the following:

```python
list1, list2
```

```
('1', '2', '3', '4'), ('a', 'b', 'c', 'd')
```