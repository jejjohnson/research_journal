# Dictionaries to Lists



??? info "Full Function"

    ```python
    """
    NOTE! Need the mypy (typing) package
    >>> !pip install mypy
    """
    import itertools
    from typing import Dict, List


    def dict_product(dicts: Dict) -> List[Dict]:
        """Returns the product of a dictionary with lists
        Parameters
        ----------
        dicts : Dict,
            a dictionary where each key has a list of inputs
        Returns
        -------
        prod : List[Dict]
            the list of dictionary products
        Example
        -------
        >>> parameters = {
            "samples": [100, 1_000, 10_000],
            "dimensions": [2, 3, 10, 100, 1_000]
            }
        >>> parameters = list(dict_product(parameters))
        >>> parameters
        [{'samples': 100, 'dimensions': 2},
        {'samples': 100, 'dimensions': 3},
        {'samples': 1000, 'dimensions': 2},
        {'samples': 1000, 'dimensions': 3},
        {'samples': 10000, 'dimensions': 2},
        {'samples': 10000, 'dimensions': 3}]
        """
        return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))
    ```

**Resources**:

* [Using itertools.product with dictionaries](http://stephantul.github.io/python/2019/07/20/product-dict/) - Stephan Tulkens
* [Using itertools.product instead of nested for loops](https://stephantul.github.io/python/2019/07/20/product/) - Stephan Tulkens