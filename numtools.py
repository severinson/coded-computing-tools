'''This module provides tools for numerical computations used by the
other modules. In particular, it provides a function for numerically
evaluating the inverse of a cumulative distribution function through
binary search.

'''

import math

def numerical_inverse(fun, value, min_value, max_value, eps=0.00000001, discrete=True):
    '''Numerically evaluate the inverse of a function by performing binary
    search.

    Args:

    fun: A function for which there exists an inverse.

    value: A value to evaluate inverse at.

    min_value: The minimum value for which fun is defined.

    max_value: The maximum value for which fun is defined.

    eps: Stop when the current value is within eps of the requested
    value. Not guaranteed if discrete is True.

    discrete: Wheter fun is only defined for integer values only.

    Returns: ICDF(value).

    '''
    assert callable(fun)
    assert isinstance(value, float)
    assert isinstance(discrete, bool)
    if discrete:
        assert isinstance(min_value, int)
        assert isinstance(max_value, int)
    else:
        isinstance(min_value, float)
        isinstance(max_value, float)
    assert min_value > - math.inf
    assert max_value < math.inf
    assert min_value < max_value
    assert isinstance(eps, float)
    assert eps > 0

    while True:
        current_value = min_value + (max_value - min_value) / 2
        if discrete:
            current_value = round(current_value)

        fun_value = fun(current_value)
        if abs(fun_value - value) < eps:
            break

        if discrete and max_value - min_value <= 1:
            if abs(fun(min_value) - value) < abs(fun(max_value) - value):
                current_value = min_value
            else:
                current_value = max_value
            break

        if fun_value < value:
            min_value = current_value
        else:
            max_value = current_value

    return current_value
