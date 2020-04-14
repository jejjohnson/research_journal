# Parallel Processing


## Multiprocessing Module

=== info "Full Function"

    ```python
    from joblib import Parallel, delayed
    from typing import Callable, Iterable

    def run_parallel_step(
        exp_step: Callable,
        parameters: Iterable,
        n_jobs: int = 2,
        verbose: int = 1,
        **kwargs
    ) -> List:
        """Helper function to run experimental loops in parallel
        
        Parameters
        ----------
        exp_step : Callable
            a callable function which does each experimental step
        
        parameters : Iterable,
            an iterable (List, Dict, etc) of parameters to be looped through
        
        n_jobs : int, default=2
            the number of cores to use
        
        verbose : int, default=1
            the amount of information to display in the 
        
        Returns
        -------
        results : List
            list of the results from the function
            
        Examples
        --------
        Example 1 - No keyword arguments
        >>> parameters = [1, 10, 100]
        >>> def step(x): return x ** 2
        >>> results = run_parallel_step(
            exp_step=step,
            parameters=parameters,
            n_jobs=1, verbose=1
        )
        >>> results
        [1, 100, 10000]
        Example II: Keyword arguments
        >>> parameters = [1, 10, 100]
        >>> def step(x, a=1.0): return a * x ** 2
        >>> results = run_parallel_step(
            exp_step=step,
            parameters=parameters,
            n_jobs=1, verbose=1,
            a=10
        )
        >>> results
        [100, 10000, 1000000]
        """

        # loop through parameters
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(exp_step)(iparam, **kwargs) for iparam in parameters
        )
        return results
    ```