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


import logging
import random
import math
from fractions import Fraction
import heapq
import itertools
import numpy as np
from scipy.misc import comb as nchoosek
import model
from assignments.sparse import SparseAssignment

def weight_distribution(n, k, normalization):
    '''Compute the distribution of weight over a set of subtrees
    represented by a list. The subtree corresponding to each element
    in the list is the number of possibly batches that start with that
    element.

    Args:

    n: Number of subtrees.

    k: Number of selections.

    normalization: Normalize the weights to sum to this value.

    Returns: A list of Fractions proportional to the weight of the
    corresponding subtree, such that all weights sum to normalization.

    '''
    assert isinstance(n, int) or isinstance(n, np.int64)
    assert isinstance(k, int) or isinstance(k, np.int64)
    assert normalization % 1 == 0
    weights = np.asarray([nchoosek(i + 1, k, exact=True) for i in range(n - 1, -1, -1)])
    # weights -= np.cumsum(weights)
    total_weight = weights.sum()
    # total_weight = sum(weights)
    if total_weight == 0:
        return [0] * n

    distribution = weights / total_weight * normalization
    return distribution

def weight_distribution_slow(n, k, normalization):
    '''Compute the distribution of weight over a set of subtrees
    represented by a list. The subtree corresponding to each element
    in the list is the number of possibly batches that start with that
    element.

    Args:

    n: Number of subtrees.

    k: Number of selections.

    normalization: Normalize the weights to sum to this value.

    Returns: A list of Fractions proportional to the weight of the
    corresponding subtree, such that all weights sum to normalization.

    '''
    assert isinstance(n, int) or isinstance(n, np.int64)
    assert isinstance(k, int) or isinstance(k, np.int64)
    assert normalization % 1 == 0
    weights = np.asarray([nchoosek(i + 1, k, exact=True) for i in range(n)])
    # weights -= np.cumsum(weights)
    total_weight = weights.sum()
    if total_weight == 0:
        return [0] * n

    distribution = [float(Fraction(normalization * weight, total_weight))
                    for weight in reversed(weights)]
    return distribution

def truncate_distribution(distribution, limits, normalization):
    '''Truncate the values of a distribution by clamping each value to
    some maximum. The distribution is normalized to sum to the same
    value after clamping, and zeroes in the original distribution will
    remain zero..

    Args:

    distribution: List of values making up the distribution.

    limits: List of max values. Must be of same length as
    distribution.

    normalization: Normalize the weights to sum to this value.

    Returns: The truncated distribution.

    '''
    assert len(distribution) == len(limits), str(len(distribution)) + '!=' + str(len(limits))
    distribution = np.asarray(distribution)
    limits = np.asarray(limits)
    if limits.sum() < normalization:
        raise ValueError('Cannot normalize by {} with limits summing to {} for limits {}.'.format(normalization,
                                                                                                  limits.sum(),
                                                                                                  limits))

    clipped_distribution = np.clip(distribution, [0]*len(distribution), limits)
    if clipped_distribution.sum() < normalization:
        headroom = limits - clipped_distribution
        headroom[distribution == 0] = 0
        if not headroom.sum():
            raise ValueError('No headroom left for distribution \n%s\nlimits \n%s\nnormalization %f.',
                             str(distribution), str(limits), normalization)

        percentages = headroom / headroom.sum()
        clipped_distribution = clipped_distribution + percentages * (normalization - clipped_distribution.sum())

    return clipped_distribution

def integer_distribution(n, k, normalization, offset=0, limits=None):
    '''Distribute an integer over a set of subtrees represented by a list.
    The subtree corresponding to each element in the list is the
    number of possible batches that start with that element.

    Args:

    n: Number of subtrees.

    k: Number of selections.

    normalization: Normalize the distribution to sum to this value.

    offset: The partition index offset.

    limits: List of max values. Must be of same length as
    distribution.

    Returns: A list of tuples (index, value), with value being
    integers proportional to the weight of the corresponding subtree,
    such that all values sum to normalization.

    '''
    assert n % 1 == 0
    assert k % 1 == 0
    assert normalization % 1 == 0
    assert offset % 1 == 0 and offset >= 0
    assert k > 0
    assert n >= k

    if not normalization:
        return []

    # Assign values to elements with values >= 1.
    weights = weight_distribution(n, k, normalization)
    if limits is not None:
        weights = truncate_distribution(weights, limits[offset:], normalization)

    integers = dict()
    remaining = normalization
    for i, weight in zip(range(len(weights)), weights):
        value = math.floor(weight)
        if value <= 0:
            continue

        integers[i + offset] = value
        weights[i] -= value
        remaining -= value

    # Assign any remaining indices.
    while remaining > 0:
        index = None
        value = 0
        for i in range(n):
            if weights[i] <= 0:
                continue

            if limits[offset + i] > value:
                index = i
                value = limits[offset + i]

        weights[index] = 0
        if index + offset in integers:
            integers[index + offset] += 1
        else:
            integers[index + offset] = 1

        remaining -= 1

    return list(integers.items())

class BatchPrefix(object):
    '''Representation of a partially completed batch.

    '''

    def __init__(self, indices, remaining, partitions, children, remaining_values):
        '''Create a partial batch.

        Args:

        indices: The partition indices of the values having been
        assigned to the batch so far.

        remaining: Number of assignments remaining for this batch.

        partitions: Total number of partitions.

        children: Number of batches for which this batch is a prefix.

        remaining_values: List of remaining values to be assiged to
        each partition.

        '''
        assert isinstance(indices, list)
        assert isinstance(remaining, int) or isinstance(remaining, np.int64)
        assert remaining >= 0
        self.indices = indices
        self.indices.sort()
        if self.indices:
            self.max_index = self.indices[-1]
        else:
            self.max_index = -1
        self.remaining = remaining
        self.partitions = partitions
        if self.remaining:
            self.children = children
        else:
            self.children = children - 1

        self.remaining_values = remaining_values

        # Allocate space for children
        self.allocate()
        # try:
        #     self.allocate()
        # except ValueError:
        #     self.max_index = -1
        #     self.allocate()

        # return

    def __repr__(self):
        string = 'Indices: ' + str(self.indices)
        string += ' Remaining: ' + str(self.remaining)
        string += ' Children: ' + str(self.children)
        return string

    def allocate(self):
        '''Allocate indices for a child of this prefix.'''
        # if not self.remaining:
        #     raise ValueError('Cannot allocate with no remaining values. {}'.format(self))

        if self.remaining:
            self.distribution = integer_distribution(self.partitions - self.max_index - 1,
                                                     self.remaining, self.children,
                                                     offset=self.max_index + 1,
                                                     limits=self.remaining_values)
        else:
            self.distribution = []

        for partition, children in self.distribution:
            self.remaining_values[partition] -= children

        return

    def fork(self):
        '''Create a batch prefix that in turn has this object as its prefix.

        remaining == 0, children == 0 => None, None
        Remaining == 0, children > 0 => parent, None
        Remaining > 0, children > 0 => child, parent

        Returns: A tuple (child, parent) with the forked child and
        this object as the parent.

        '''

        if not self.children:
            return None, None

        if not self.remaining and self.children:
            self.children -= 1
            return self, None

        if self.remaining and self.children:
            remaining = self.remaining - 1
            # print(self)
            # print(self.distribution)
            partition, children = self.distribution.pop()

            self.children -= children
            indices = self.indices + [partition]
            try:
                child = BatchPrefix(indices, remaining, self.partitions,
                                    children, self.remaining_values)
            except ValueError as err:
                logging.error(err)
                self.remaining_values[partition] += children
                return self, None

            # Allocate indices for grandchildren
            # try:
            #     child.allocate()
            # except ValueError as err:
            #     logging.error(err)
            #     # child.max_index = - 1
            #     # child.allocate()
            #     return None, self

            return child, self

        raise ValueError('Unknown error when forking prefix ' + str(self))

def batch_generator(parameters):
    '''Generate batches in a way that attempts to maximize the difference
    between all pairs of batches.

    Args:

    parameters: Parameters object.

    '''
    assert isinstance(parameters, model.SystemParameters)
    rows_per_element = parameters.num_coded_rows
    rows_per_element /= parameters.num_partitions * parameters.num_batches
    rows_per_element = math.floor(rows_per_element)

    remaining_rows_per_batch = parameters.rows_per_batch
    remaining_rows_per_batch -= rows_per_element * parameters.num_partitions
    remaining_rows_per_batch = round(remaining_rows_per_batch)

    remaining_rows_per_partition = parameters.num_coded_rows / parameters.num_partitions
    remaining_rows_per_partition -= rows_per_element * parameters.num_batches
    remaining_rows_per_partition = round(remaining_rows_per_partition)

    remaining_values = [remaining_rows_per_partition] * parameters.num_partitions
    prefixes = [BatchPrefix([], remaining_rows_per_batch,
                            parameters.num_partitions,
                            parameters.num_batches,
                            remaining_values)]

    completed = 0
    while prefixes:
        parent = prefixes.pop()
        child, parent = parent.fork()
        logging.debug('Completed {} batches.\nParent: {}\nChild: {}\n'.format(completed, parent, child))

        if parent:
            prefixes.append(parent)

        if child:
            prefixes.append(child)

        if child and not child.remaining:
            completed += 1
            yield child.indices

    # One batch is sometimes missing due to numerical instability.
    # This takes care of adding this batch.
    if completed == parameters.num_batches - 1:
        yield list(range(remaining_rows_per_batch))

    if completed - parameters.num_batches:
        print(completed - parameters.num_batches, 'REMAINING', remaining_values, sum(remaining_values), remaining_rows_per_batch)

    return

class TreeSolver(object):
    '''This solver creates an assignment using a heuristic block-diagonal
    structure.

    '''

    def __init__(self):
        return

    def solve(self, parameters):
        '''Create an assignment using a block-diagonal structure.

        Args:
        parameters: System parameters

        verbose: Print extra messages if True.

        Returns: The resulting assignment

        '''
        assert isinstance(parameters, model.SystemParameters)
        rows_per_element = parameters.num_coded_rows
        rows_per_element /= parameters.num_partitions * parameters.num_batches
        rows_per_element = math.floor(rows_per_element)
        assignment = SparseAssignment(parameters, gamma=rows_per_element)

        rows = list()
        cols = list()
        data= list()

        # Assign the remaining rows, attempting to minimize repetition
        for batch, row in zip(batch_generator(parameters), range(parameters.num_batches)):
            logging.debug('Got batch %s.', str(batch))

            data += [1] * len(batch)
            rows += [row] * len(batch)
            cols += batch

        assignment.increment(rows, cols, data)
        return assignment

    @property
    def identifier(self):
        '''Return a string identifier for this object.'''
        return self.__class__.__name__
