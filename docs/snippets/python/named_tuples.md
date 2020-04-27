# Named Tuples

I've recently grown a liking to `namedtuple`'s. It's a nice way to have custom containers.

```python
from collections import namedtuple
```

---

## Instance of

```python
your_tuple = namedtuple('your_tuple', ['entry1', 'entry2'])

isinstance(a, your_tuple)
```
