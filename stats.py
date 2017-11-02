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
import statistics
import numpy as np
import scipy as sp
import scipy.stats

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
    samples = [order_sample(icdf, total, order) for _ in range(samples)]
    mean = statistics.mean(samples)
    variance = statistics.variance(samples, xbar=mean)
    return mean, variance

@functools.lru_cache(maxsize=1024)
def order_mean_shiftexp(total, order, parameter=1):
    '''Compute the mean of the shifted exponential order statistic.

    Args:

    total: Total number of variables.

    order: Statistic order.

    parameter: Distribution parameter.

    Returns: The mean of the order statistic with variables drawn from
    the shifted exponential distribution.

    '''
    mean = 1
    for i in range(total-order+1, total+1):
        mean += 1 / i

    mean *= parameter
    return mean

@functools.lru_cache(maxsize=1024)
def order_variance_shiftexp(total, order, parameter):
    '''Compute the variance of the shifted exponential order statistic.

    Args:

    total: Total number of variables.

    order: Statistic order.

    parameter: Distribution parameter.

    Returns: The variance of the order statistic with variables drawn from
    the shifted exponential distribution.

    '''
    variance = 0
    for i in range(total-order+1, total+1):
        variance += 1 / math.pow(i, 2)

    variance *= math.pow(parameter, 2)
    return variance

class ShiftexpOrder(object):
    '''Shifted exponential order statistic random variable.'''

    def __init__(self, parameter, total, order):
        '''Initialize the object.

        Args:

        parameter: Shifted exponential parameter.

        '''
        assert 0 < parameter < math.inf
        assert 0 < total < math.inf and total % 1 == 0
        assert 0 < order <= total and order % 1 == 0
        self.parameter = parameter
        self.total = total
        self.order = order
        self.beta = self.mean() / self.variance()
        self.alpha = self.mean() * self.beta
        return

    def pdf(self, value):
        '''Probability density function.'''
        assert 0 <= value <= math.inf
        return scipy.stats.gamma.pdf(value, self.alpha, scale=1/self.beta,
                                     loc=self.parameter)

    def mean(self):
        '''Expected value.'''
        return order_mean_shiftexp(self.total, self.order,
                                   parameter=self.parameter,
                                   exact=True)

    def variance(self):
        '''Random variable variance.'''
        return order_variance_shiftexp(self.total, self.order,
                                       self.parameter)

class Shiftexp(object):
    '''Shifted exponential distributed random variable.'''

    def __init__(self, parameter):
        '''Initialize the object.

        Args:

        parameter: Shifted exponential parameter.

        '''
        assert 0 < parameter < math.inf
        self.parameter = parameter
        return

    def pdf(self, value):
        '''Shifted exponential distribution PDF.'''
        assert 0 <= value <= math.inf
        if value < self.parameter:
            return 0
        else:
            return 1 / self.parameter * math.exp(-(value / self.parameter - 1))

    def cdf(self, value):
        '''Shifted exponential distribution CDF.'''
        assert 0 <= value <= math.inf
        if value < self.parameter:
            return 0
        else:
            return 1 - math.exp(-(value / self.parameter - 1))

    def icdf(self, value):
        '''Shifted exponential distribution inverse CDF.'''
        assert 0 <= value < 1, 'Value must be <= 0 and < 1.'
        return self.parameter * (1 - math.log(1 - value))

    def mean(self):
        '''Expected value.'''
        return self.parameter

def validate():
    '''Validate the analytic computation of the order stats.'''
    total = 1000
    order = 900
    mu = 2
    icdf = lambda x: icdf_shiftexp(x, parameter=mu)
    mean, variance = order_mean_empiric(icdf, total, order, samples=10000)

    # samples = [order_sample(icdf, total, order) for _ in range(100000)]
    # mean = sum(samples) / len(samples)
    analytic_mean = order_mean_shiftexp(total, order, parameter=mu)
    analytic_variance = order_variance_shiftexp(total, order, parameter=mu)

    try:
        fast_analytic = order_mean_shiftexp(total, order, parameter=mu, exact=False)
    except (ValueError, NotImplementedError):
        fast_analytic = math.inf
    print(mean, analytic_mean)
    print(variance, analytic_variance)
    return

if __name__ == '__main__':
    validate()
