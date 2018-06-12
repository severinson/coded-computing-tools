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
import matplotlib.pyplot as plt

from scipy.special import gamma, gammainc

def order_sample(icdf=None, total=None, order=None):
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
    return partition[order-1]

def order_samples(icdf=None, total=None, order=None, samples=None):
    '''Sample the order statistic of the distribution given by the icdf'''
    return [order_sample(icdf, total, order) for _ in range(samples)]

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
    if total % 1 != 0:
        raise ValueError("total={} must be an integer".format(total))
    if order % 1 != 0:
        raise ValueError("order={} must be an integer".format(order))
    total, order = int(total), int(order)
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

@functools.lru_cache(maxsize=1024)
def order_cdf_shiftexp(value, total=None, order=None, parameter=None):
    '''CDF of the shifted exponential order statistic.

    args:

    value: value to evaluate the CDF at.

    total: Total number of variables.

    order: Statistic order.

    parameter: Distribution parameter.

    returns: the CDF of the shifted exponential order statistic evaluated at
    value.

    '''
    assert 0 <= value <= math.inf
    if value < parameter:
        return 0
    a = order_mean_shiftexp(total, order, parameter) - parameter
    a /= order_variance_shiftexp(total, order, parameter)
    b = pow(order_mean_shiftexp(total, order, parameter) - parameter, 2)
    b /= order_variance_shiftexp(total, order, parameter)

    # the CDF is given by the regularized lower incomplete gamma function.
    return gammainc(b, a*(value-parameter))

def order_aggregate_cdf_shiftexp(value, parameter=None, total=None,
                                 orders=None, order_probabilities=None):
    '''aggregate CDF of the shifted exponential order statistic.

    this method aggregates the order statistic CDF when the order is a random
    variable.

    args:

    value: value to evaluate the aggregate CDF at.

    parameter: shifted exponential distribution parameter.

    total: total number of variables.

    orders: numpy array-like of orders to aggregate the CDF from.

    order_probabilities: numpy array-like with probabilities for each order
    being active.

    returns: the aggregate CDF evaluated at value.

    '''
    assert len(orders) == len(order_probabilities)
    assert np.allclose(order_probabilities.sum(), 1)

    # evaluate the shifted exponential order statistic CDF at each order
    cdf_values = np.fromiter(
        (order_cdf_shiftexp(value, total=total, order=i, parameter=parameter) for i in orders),
        dtype=float,
    )

    # weigh the results by the probability of the respective order
    cdf_values *= order_probabilities

    # summing the array gives the weighted average, which is the value of the
    # aggregate CDF.
    return cdf_values.sum()

class ShiftexpOrder(object):
    '''Shifted exponential order statistic random variable.'''

    def __init__(self, parameter=None, total=None, order=None):
        '''Initialize the object.

        Args:

        parameter: Shifted exponential parameter.

        '''
        assert 0 < parameter < math.inf
        assert 0 < total < math.inf and total % 1 == 0
        assert 0 <= order <= total and order % 1 == 0
        self.parameter = parameter
        self.total = total
        self.order = order

        # this variable is gamma distributed. compute the distribution
        # parameters from its mean and variance.
        self.a = (self.mean()-self.parameter) / self.variance()
        self.b = pow(self.mean()-self.parameter, 2) / self.variance()
        return

    def pdf(self, value):
        '''Probability density function.'''
        assert 0 <= value <= math.inf
        return scipy.stats.gamma.pdf(
            value,
            self.b,
            scale=1/self.a,
            loc=self.parameter,
        )

    def cdf(self, value):
        '''cumulative distribution function'''
        return order_cdf_shiftexp(
            value=value,
            total=self.total,
            order=self.order,
            parameter=self.parameter,
        )

    def mean(self):
        '''Expected value.'''
        if self.order == 0:
            return 0.0
        return order_mean_shiftexp(
            self.total,
            self.order,
            parameter=self.parameter,
        )

    def variance(self):
        '''Random variable variance.'''
        return order_variance_shiftexp(
            self.total,
            self.order,
            self.parameter,
        )

    def sample(self, n=1):
        '''sample the distribution'''
        return scipy.stats.gamma.rvs(
            self.b,
            scale=1/self.a,
            loc=self.parameter,
            size=n,
        )

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

class ExpSum(object):
    '''Random variable representing the sum of order RVs with exponential
    distribution and given parameter.

    '''

    def __init__(self, scale=None, order=None):
        self.b = order
        self.scale = scale
        return

    def mean(self):
        return self.scale*self.order

    def pdf(self, value):
        return scipy.stats.gamma.pdf(value, self.b, scale=self.scale, loc=0)

    def cdf(self, value):
        return scipy.stats.gamma.cdf(value, self.b, scale=self.scale, loc=0)

def validate():
    '''Validate the analytic computation of the order stats.'''
    total = 9
    order = 6
    num_samples = 100000
    mu = 2
    x = Shiftexp(mu)
    x_order = ShiftexpOrder(mu, total, order)
    samples = order_samples(icdf=x.icdf, total=total, order=order, samples=num_samples)
    samples_gamma = x_order.sample(n=num_samples)
    t = np.linspace(min(samples), max(samples), 100)

    plt.figure()
    plt.hist(samples, bins=200, histtype='stepfilled', alpha=0.7, density=True)
    plt.hist(samples_gamma, bins=200, histtype='stepfilled', alpha=0.7, density=True)
    plt.grid()

    plt.figure()
    plt.hist(samples, bins=200, histtype='stepfilled', density=True)
    pdf = [x_order.pdf(i) for i in t]
    plt.plot(t, pdf)
    plt.grid()

    plt.figure()
    plt.hist(samples, bins=100, cumulative=True, density=True)
    cdf = [x_order.cdf(i) for i in t]
    plt.plot(t, cdf)
    plt.grid()

    plt.figure()
    plt.hist(samples, bins=100, cumulative=True, density=True)
    cdf = [order_aggregate_cdf_shiftexp(
        i,
        parameter=mu,
        total=total,
        orders=np.array([order, order+1]),
        order_probabilities=np.array([0.75, 0.25])
    ) for i in t]
    plt.plot(t, cdf)
    plt.grid()
    plt.show()
    return

if __name__ == '__main__':
    validate()
