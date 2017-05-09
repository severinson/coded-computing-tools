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

'''This module provides functionality for efficiently sampling the
performance of an assignment. Specifically, the performance is
evaluated through Monte Carlo simulations by randomizing the order in
which servers finish. The number of servers needed in a given Monte
Carlo iteration is computed by performing binary search.

'''

import math
import random
import logging
import itertools
import datetime
import numpy as np
import pandas as pd
from model import SystemParameters, ModelError
from assignments import Assignment
from evaluation import AssignmentEvaluator

class SampleEvaluator(AssignmentEvaluator):
    '''This evaluator samples the performance of an assignment. It uses
    binary search to efficiently compute the delay.

    '''

    def __init__(self, num_samples=1000):
        '''Create a sample evaluator.

        Args:

        num_samples: Number of performance samples to take.

        '''
        assert isinstance(num_samples, int) and num_samples > 0
        self.num_samples = num_samples
        return

    def evaluate(self, parameters, assignment):
        '''Sample the communication load and computational delay of an
        assignment.

        Args:

        parameters: System parameters

        assignment: Assignment to evaluate.

        Returns: Dict with entries for estimated number of unicasts,
        and estimated number of servers required to wait for.

        '''
        assert isinstance(parameters, SystemParameters), type(parameters)
        assert isinstance(assignment, Assignment)
        results = list()
        printout_interval = datetime.timedelta(seconds=10)
        last_printout = datetime.datetime.utcnow()

        for i in range(self.num_samples):

            # Print progress periodically
            if datetime.datetime.utcnow() - last_printout > printout_interval:
                last_printout = datetime.datetime.utcnow()
                logging.info('%s %f percent finished.',
                             parameters.identifier(), i / self.num_samples * 100)

            # Randomly generate a server completion order
            completion_order = random.sample(range(parameters.num_servers), parameters.num_servers)

            # Evaluate it
            result = dict()
            result.update(computational_delay_sample(parameters, assignment, completion_order))
            result.update(communication_load_sample(parameters, assignment, completion_order))
            results.append(result)

        return pd.DataFrame(results)

def decodeable(parameters, assignment, batches,
               incomplete_partitions, permanent_count):
    '''Evaluate if decoding is possible for a given set of batches.

    Args:

    parameters: Parameters object

    assignment: Assignment object

    batches: The set of batches to add to the total count.

    incomplete_partitions: A set of batches that are not yet decodeable.

    permanent_count: A Numpy array of length prameters.num_partitions
    with counts[i] indicating the number of elements stored from
    partition i.

    '''
    assert isinstance(parameters, SystemParameters)
    assert isinstance(assignment, Assignment)
    assert isinstance(batches, set)
    assert isinstance(incomplete_partitions, set)

    tentative_count = permanent_count + assignment.batch_union(batches)
    remaining = (tentative_count < parameters.rows_per_partition).sum()
    if remaining > 0:
        return False, tentative_count
    else:
        return True, tentative_count

def computational_delay_sample(parameters, assignment, completion_order):
    '''Compute the computational delay of one realization of the server
    completion order.

    Args:

    parameters: System parameters.

    assignment: Assignment to evaluate.

    completion_order: A list of server indices in the order they
    completed their map phase computation.

    Returns: A dict containing the results.

    '''

    # Store the batches added permanently
    permanently_added = set()
    for server in completion_order[0:parameters.q]:
        permanently_added.update(assignment.labels[server])

    # TODO: Use a leaner dtype?
    permanent_count = np.zeros(parameters.num_partitions)
    permanent_count += assignment.batch_union(permanently_added)

    incomplete_partitions = set(range(parameters.num_partitions))

    # Find the number of servers through binary search.
    min_bound = parameters.q
    max_bound = parameters.num_servers
    while min_bound < max_bound:
        current = min_bound + math.floor((max_bound - min_bound) / 2)

        # Add servers
        tentatively_added = set()
        for server in completion_order[min_bound:current]:
            tentatively_added.update(assignment.labels[server])

        can_decode, tentative_count = decodeable(parameters, assignment, tentatively_added,
                                                 incomplete_partitions, permanent_count)
        if can_decode:
            max_bound = current
        else:
            min_bound = current + 1
            permanent_count = tentative_count
            permanently_added.update(tentatively_added)

    coded_rows_per_server = parameters.num_source_rows * parameters.server_storage
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    return {'servers': min_bound, 'batches': min_bound * batches_per_server,
            'delay': parameters.computational_delay(q=min_bound)}

def communication_load_sample(parameters, assignment, completion_order):
    '''Compute the communication load of one realization of the server
    completion order.

    Args:

    parameters: System parameters.

    assignment: Assignment to evaluate.

    completion_order: A list of server indices in the order they
    completed their map phase computation.

    Returns: A dict containing the results.

    '''

    # Only the first q servers are used for multicasting.
    completion_order = completion_order[0:parameters.q]

    # Select 1 server and compute the load from its perspective.
    server = completion_order[-1]

    # Select the remaining q - 1 servers
    servers_without_q = completion_order[0:-1]
    assert len(completion_order) == parameters.q

    # Sum the corresponding rows of the assignment matrix
    batches_1 = assignment.labels[server].copy()

    # Multicasting load
    multicast_load_1, multicast_load_2 = parameters.multicast_load()

    # Strategy 1 multicasts for all subsets of size from sq to muq.
    try:
        for j in range(parameters.multicast_set_size_1(), parameters.muq + 1):
            for multicast_subset in itertools.combinations(servers_without_q, j):
                batches_1.update(set.intersection(*[assignment.labels[x]
                                                    for x in multicast_subset]))
    except ModelError:
        pass

    # Negative values indicate remaining needed values
    count_vector = np.zeros(parameters.num_partitions)
    count_vector -= parameters.num_source_rows / parameters.num_partitions

    # Add multicasted values
    count_vector += assignment.batch_union(batches_1)

    # Compute unicasts by summing the negative elements
    unicast_load_1 = abs(count_vector[count_vector < 0].sum())
    unicast_load_1 /= parameters.num_source_rows

    # Strategy 2 multicasts for subsets of size sq-1 to muq
    batches_2 = set()
    try:
        for multicast_subset in itertools.combinations(servers_without_q,
                                                       parameters.multicast_set_size_2()):
            batches_2.update(set.intersection(*[assignment.labels[x]
                                                for x in multicast_subset]))
    except ModelError:
        pass

    # Add the new unique batches and compute the strategy 2 unicasts.
    count_vector += assignment.batch_union(batches_2 - batches_1)
    unicast_load_2 = abs(count_vector[count_vector < 0].sum())
    unicast_load_2 /= parameters.num_source_rows

    # Scale unicasts by number of outputs.
    unicast_load_1 *= parameters.num_outputs
    unicast_load_2 *= parameters.num_outputs

    # Append the results.
    return {'unicast_load_1': unicast_load_1, 'unicast_load_2': unicast_load_2,
            'multicast_load_1': multicast_load_1, 'multicast_load_2': multicast_load_2}
