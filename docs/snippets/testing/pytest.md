---
title: PyTest Tricks
description: The deep learning python stack
authors:
    - J. Emmanuel Johnson
path: docs/snippets/testing
source: pytest.md
---


## Annotating Tests

```python
import pytest

@pytest.mark.slow
def test_that_runs_slowly():
    ...

@pytest.mark.data
def test_that_goes_on_data():
    ...

@pytest.mark.slow
@pytest.mark.data
def test_that_goes_on_data_slowly():
    ...
```

```bash
> py.test -m "slow"
> py.test -m "data"
> py.test -m "not data"
```

**Source**: Eric Ma - [Blog](https://ericmjl.github.io/blog/2018/2/25/annotating-code-tests-and-selectively-running-tests/)