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
import model
import assignments

def evaluate(par, assignment, num_samples=1000):
    '''Return the estimated communication load and computational delay.

    Estimate the number of unicasts required for all servers to hold
    enough symbols to deocde all partitions for some assignment
    through Monte Carlo simulations.

    Args:
    par: System parameters

    assignment: Assignment object

    num_samples: Number of runs

    Returns: Dict with entries for estimated number of unicasts, and
    estimated number of servers required to wait for.

    '''
    assert isinstance(par, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(num_samples, int)
    result = dict()
    result.update(computational_delay(par, assignment, num_samples=num_samples))
    result.update(communication_load(par, assignment, num_samples=num_samples))
    return result

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
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(batches, set)
    assert isinstance(incomplete_partitions, set)

    tentative_count = permanent_count + assignment.batch_union(batches)
    if tentative_count.sum() < parameters.num_partitions * parameters.rows_per_partition:
        return False, tentative_count

    complete_partitions = {partition for partition in incomplete_partitions
                           if tentative_count[partition] >= parameters.rows_per_partition}

    if len(incomplete_partitions) == len(complete_partitions):
        return True, tentative_count
    else:
        incomplete_partitions -= complete_partitions
        return False, tentative_count

def computational_delay(parameters, assignment, num_samples=1000):
    '''Sample the computational delay of an assignment.

    Estimate the number of servers required to decode all partitions
    for an assignment through Monte Carlo simulations.

    Args:

    parameters: System parameters

    assignment_matrix: An Assignment object.

    num_runs: Number of runs

    Returns: Average Number of servers required.

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(num_samples, int)
    result = {'batches': 0, 'servers': 0, 'delay': 0}

    printout_interval = datetime.timedelta(seconds=10)
    last_printout = datetime.datetime.utcnow()

    coded_rows_per_server = parameters.num_source_rows * parameters.server_storage
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    for i in range(num_samples):

        # Print progress periodically
        if datetime.datetime.utcnow() - last_printout > printout_interval:
            last_printout = datetime.datetime.utcnow()
            logging.info('%s %f percent finished.', parameters.identifier(), i / num_samples * 100)

        # Randomize the server completion order
        server_order = random.sample(range(parameters.num_servers), parameters.num_servers)

        # Store the batches added permanently
        permanently_added = set()
        for server in server_order[0:parameters.q]:
            permanently_added.update(assignment.labels[server])

        permanent_count = np.zeros(parameters.num_partitions)
        permanent_count += assignment.batch_union(permanently_added)

        incomplete_partitions = set(range(parameters.num_partitions))

        min_bound = parameters.q
        max_bound = parameters.num_servers
        while min_bound < max_bound:
            current = min_bound + math.floor((max_bound - min_bound) / 2)

            # Add servers
            tentatively_added = set()
            for server in server_order[min_bound:current]:
                tentatively_added.update(assignment.labels[server])

            can_decode, tentative_count = decodeable(parameters, assignment, tentatively_added,
                                                     incomplete_partitions, permanent_count)
            if can_decode:
                max_bound = current
            else:
                min_bound = current + 1
                permanent_count = tentative_count
                permanently_added.update(tentatively_added)

        result['servers'] += min_bound
        result['batches'] += min_bound * batches_per_server
        result['delay'] += parameters.computational_delay(q=min_bound)

    result['servers'] /= num_samples
    result['batches'] /= num_samples
    result['delay'] /= num_samples
    return result

def communication_load(parameters, assignment, num_samples):
    '''Sample the communication load of an assignment.

    Estimate the number of unicasts required for all servers to hold
    enough symbols to deocde all partitions for some assignment
    through Monte Carlo simulations.

    Args:

    parameters: System parameters

    assignment_matrix: An Assignment object.

    num_runs: Number of runs

    Returns: Average number of unicasts required.

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(num_samples, int)

    result = {'load': 0}
    server_list = list(range(parameters.num_servers))

    for _ in range(num_samples):

        # Generate a random set of finished servers and select one
        finished_servers = random.sample(server_list, parameters.q)
        finished_server = random.sample(finished_servers, 1)[0]

        # Sum the corresponding rows of the assignment matrix
        batches = set()
        for batch in assignment.labels[finished_server]:
            batches.add(batch)

        # Add the rows sent in the shuffle phase
        for j in range(parameters.sq, int(parameters.server_storage*parameters.q) + 1):
            for multicast_subset in itertools.combinations([x for x in finished_servers
                                                            if x != finished_server], j):
                batches.update(set.intersection(*[assignment.labels[x] for x in multicast_subset]))

        count_vector = np.zeros(parameters.num_partitions) - parameters.num_source_rows / parameters.num_partitions
        count_multicasts = assignment.batch_union(batches)
        count_vector = np.add(count_vector, count_multicasts)
        result['load'] += model.remaining_unicasts(count_vector)

    result['load'] *= parameters.q
    result['load'] /= num_samples
    return result
