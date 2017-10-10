'''This module provides tools for numerical computations used by the
other modules. In particular, it provides a function for numerically
evaluating the inverse of a cumulative distribution function through
binary search.

'''

import math

from functools import partial

def _find_limits(fun=None, target=None):
    '''bound the input needed to reach value'''
    assert callable(fun)
    assert target is not None
    value = 1
    while fun(value) < target:
        value *= 2
    return math.floor(value / 2), value

def _bounds_wrap(value, fun=None, lower=None, upper=None):
    '''wrap a function with bounds checking'''
    if value < lower:
        return -math.inf
    if value > upper:
        return math.inf
    return fun(value)

def numinv(fun=None, target=None, lower=None, upper=None):
    '''numerically invert a discrete function

    args:

    fun: function to invert. must be monotonically increasing in the range
    lower to upper. use lambdas to wrap your function if this is not the case.

    target: target value

    lower: lower bound to search, < math.inf or None to find automatically.

    upper: upper bound to search, >= 0, or None to find automatically.

    returns: value such that fun(value) <= target < fun(value+1).

    '''
    assert callable(fun)
    assert target is not None
    assert 0 <= target < math.inf
    if lower is None:
        lower = 0
    if upper is None:
        upper = math.inf

    # wrap fun with bounds checking
    ffun = partial(_bounds_wrap, fun=fun, lower=lower, upper=upper)

    # find upper and lower bounds
    lower, upper = _find_limits(ffun, target)

    # binary search
    while upper - lower > 1:
        middle = math.floor(lower + (upper - lower) / 2)
        if ffun(middle) >= target:
            upper = middle
        else:
            lower = middle

    # return whichever is closest and lower if it is a tie
    if abs(ffun(lower) - target) <= abs(ffun(upper) - target):
        return lower
    else:
        return upper

def numerical_inverse(fun, value, min_value, max_value, eps=0.00000001, discrete=True):
    '''Legacy code. Use the above numinv function instead.

    Numerically evaluate the inverse of a function by performing binary
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
