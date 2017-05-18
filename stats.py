############################################################################
# Copyright 2016 Albin Severinson                                          #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

'''Project statistics module.'''

import math
import random
import functools
import numpy as np

def order_sample(icdf, total, order):
    '''Sample the order statistic.

    Args:

    icdf: Distribution ICDF function.

    total: Total number of realizations.

    order: Order of the statistic.

    Returns: A sample of the order statistic.

    '''
    assert isinstance(total, int)
    assert isinstance(order, int)
    assert total >= order, 'order must be less or equal to total.'
    samples = [icdf(random.random()) for _ in range(total)]
    partition = np.partition(np.asarray(samples), order-1)
    # assert partition[order-1] <= partition[order]
    # assert partition[order] <= partition[order+1]
    return partition[order-1]

def order_mean_empiric(icdf, total, order, samples=1000):
    '''Compute the order static mean numerically.

    Args:

    icdf: Distribution ICDF function.

    total: Total number of realizations.

    order: Order of the statistic.

    samples: Samples to compute the mean from.

    Returns: The mean of the order statistic with variables drawn from
    the given distribution.

    '''
    samples = (order_sample(icdf, total, order) for _ in range(samples))
    return sum(samples) / len(samples)

@functools.lru_cache(maxsize=1024)
def order_mean_shiftexp(total, order, parameter=1, exact=True):
    '''Compute the mean of the shifted exponential order statistic.

    Args:

    total: Total number of variables.

    order: Statistic order.

    parameter: Distribution parameter.

    exact: Compute the mean exactly. Set to Falset to speed up the
    computation.

    Returns: The mean of the order statistic with variables drawn from
    the shifted exponential distribution.

    '''
    if exact:
        mean = 1
        for i in range(total-order+1, total+1):
            mean += 1 / i

        mean *= parameter
        return mean
    else:
        if total == order:
            raise ValueError('Non-exact computation required order < total.')
        return 1 + (1 / parameter) * math.log(total / (total - order))

def icdf_shiftexp(value, parameter=1):
    '''ICDF of the shifted exponential distribution.

    Pass uniformly distributed random numbers into this function to
    generate random numbers from the shifted exponential distribution.

    Args:

    value: The value to compute the ICDF of. Must be 0 <= value < 1.

    parameter: Distribution parameter.

    Returns:
    The ICDF evaluated at value.

    '''
    assert 0 <= value < 1, 'Value must be <= 0 and < 1.'
    return parameter * (1 - math.log(1 - value))

def validate():
    '''Validate the analytic computation of the order stats.'''
    total = 100
    order = 100
    mu = 2
    icdf = lambda x: icdf_shiftexp(x, parameter=mu)
    samples = [order_sample(icdf, total, order) for _ in range(100000)]
    mean = sum(samples) / len(samples)
    exact_analytic = order_mean_shiftexp(total, order, parameter=mu)
    try:
        fast_analytic = order_mean_shiftexp(total, order, parameter=mu, exact=False)
    except ValueError:
        fast_analytic = math.inf
    print(mean, exact_analytic, fast_analytic)
    return

if __name__ == '__main__':
    validate()
