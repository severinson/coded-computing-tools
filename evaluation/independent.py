'''This module provides tools for evaluating the performance of
assignments satisfying the following criteria:

- No element of the assignment matrix has a value larger than gamma + 1.

- The partition indices with values of gamma + 1 are independent of
  each other for all batches.

'''

import math
import random
import pandas as pd
from scipy.misc import comb as nchoosek
import numtools
import model

def decodeable_probability(num_partition_batches, total_batches,
                           num_selected_batches, rows_per_element,
                           needed_values):
    '''Compute the probability of being able to decode a given partition
    after collecting a random set of partition of some cardinality.

    Args:

    num_partition_batches: Number of batches that store a value from
    the partition to compute the decoding probability for.

    total_batches: The total number of batches.

    num_selected_batches: The number of collected batches.

    rows_per_elemenr: The number of rows assigned to all elements of
    the assignment matrix.

    needed_values: The number of values needed to decode.

    Returns: A value between 0 and 1 expressing the probability of
    being able to decode.

    '''
    assert isinstance(num_partition_batches, int)
    assert num_partition_batches >= 0
    assert isinstance(total_batches, int)
    assert total_batches > 0
    assert num_partition_batches <= total_batches
    assert isinstance(num_selected_batches, int)
    assert num_selected_batches >= 0
    assert isinstance(rows_per_element, int)
    assert rows_per_element >= 0
    assert isinstance(needed_values, int)
    assert needed_values > 0

    min_needed_batches = math.ceil(needed_values / (rows_per_element + 1))
    max_other_batches = min(needed_values, total_batches - num_partition_batches)
    max_partition_batches = math.ceil((needed_values - max_other_batches) / (rows_per_element + 1))
    max_needed_batches = max_other_batches + max_partition_batches

    # max_needed_batches = total_batches
    # extra_partition_batches = num_partition_batches * (rows_per_element + 1)
    # extra_partition_batches += (total_batches - num_partition_batches) * rows_per_element
    # extra_partition_batches -= needed_values
    # extra_partition_batches /= rows_per_element + 1
    # extra_partition_batches = min(extra_partition_batches, num_partition_batches)
    # max_needed_batches -= extra_partition_batches

    # if rows_per_element and extra_partition_batches == num_partition_batches:
    #     # extra_other_batches = (num_partition_batches - extra_partition_batches) * (rows_per_element + 1)
    #     extra_other_batches = (total_batches - num_partition_batches) * rows_per_element
    #     extra_other_batches -= needed_values
    #     extra_other_batches /= rows_per_element
    #     # extra_other_batches = min(extra_other_batches, total_batches - num_partition_batches)
    #     max_needed_batches -= extra_other_batches

    # print('min', min_needed_batches, 'max', max_needed_batches, 'selected', num_selected_batches)
    # print('extra partition', extra_partition_batches, 'partition batches', num_partition_batches)#, 'extra other', extra_other_batches)
    if num_selected_batches < min_needed_batches:
        return 0

    if num_selected_batches >= max_needed_batches:
        return 1

    result = 0
    for i in range(min_needed_batches, min(num_partition_batches, num_selected_batches) + 1):
        val = nchoosek(num_partition_batches, i, exact=True)
        val *= nchoosek(total_batches - num_partition_batches,
                        num_selected_batches - i, exact=True)
        result += val

    result /= nchoosek(total_batches, num_selected_batches, exact=True)
    return result

def estimate_delay(parameters, num_samples=100):
    '''Estimate the computational delay of an assignments meeting the
    requirements of this module.

    Args:

    parameters: Parameters object.

    num_samples: Number of samples to simulate.

    Returns: A dict with estimated required number of batches,
    servers, and delay.

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(num_samples, int) and num_samples > 0
    rows_per_element = parameters.num_coded_rows
    rows_per_element /= parameters.num_partitions * parameters.num_batches
    rows_per_element = math.floor(rows_per_element)

    remaining_rows_per_batch = parameters.rows_per_batch
    remaining_rows_per_batch -= rows_per_element * parameters.num_partitions
    remaining_rows_per_batch = round(remaining_rows_per_batch)

    remaining_rows_per_partition = parameters.num_coded_rows / parameters.num_partitions
    remaining_rows_per_partition -= rows_per_element * parameters.num_batches
    remaining_rows_per_partition = round(remaining_rows_per_partition)
    cdf = lambda x: decodeable_probability(remaining_rows_per_partition,
                                      parameters.num_batches, x,
                                      rows_per_element,
                                      parameters.rows_per_partition)

    # Sample from this distribution
    batches = list()
    for _ in range(num_samples):
        needed_batches = [numtools.numerical_inverse(cdf, random.random(), 0, parameters.num_batches)
                          for _ in range(parameters.num_partitions)]
        batches.append((max(needed_batches) - 1) * parameters.muq + 1)

    coded_rows_per_server = round(parameters.num_source_rows * parameters.server_storage)
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    servers = [math.ceil(num_batches / batches_per_server) for num_batches in batches]
    delay = [parameters.computational_delay(q=num_servers) for num_servers in servers]
    return {'batches': sum(batches) / len(batches),
            'servers': sum(servers) / len(servers),
            'delay': sum(delay) / len(delay)}

def estimate(parameters, num_samples=100):
    '''Estimate the performance of an assignment meeting the requirements of this module.

    Args:

    parameters: Parameters object.

    Returns: A Pandas DataFrame with estimated required number of
    batches, servers, and delay. Load is set to infinity as it's not
    estimated.

    '''
    result = {'load': math.inf}
    result.update(estimate_delay(parameters))
    return pd.DataFrame([result])
