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
from scipy.misc import comb as nchoosek
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
    results = list()

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

        results.append({'servers': min_bound, 'batches': min_bound * batches_per_server,
                        'delay': parameters.computational_delay(q=min_bound)})

    return pd.DataFrame(results)

def communication_load(parameters, assignment, num_samples=1000):
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
    results = list()

    for _ in range(num_samples):

        # Randomize completion order
        server_order = list(range(parameters.num_servers))
        random.shuffle(server_order)
        server_order = server_order[0:parameters.q]

        # Select 1 server and compute the load from its perspective
        server = server_order[-1]

        # Select the remaining q - 1 servers
        servers_without_q = server_order[0:-1]
        assert len(server_order) == parameters.q

        # Sum the corresponding rows of the assignment matrix
        batches = assignment.labels[server].copy()

        # Strategy 1 multicasts for all subsets of size from sq to muq
        load_1_multicasts = 0
        muq = int(parameters.server_storage * parameters.q)
        for j in range(parameters.sq, muq + 1):
            load_1_multicasts += parameters.B_j(j) / j

        load_1_multicasts *= parameters.num_source_rows * parameters.num_outputs

        for j in range(parameters.sq, muq + 1):
            for multicast_subset in itertools.combinations(servers_without_q, j):
                # load_1_multicasts += j
                batches.update(set.intersection(*[assignment.labels[x]
                                                  for x in multicast_subset]))

        # Negative values indicate remaining needed values
        count_vector = np.zeros(parameters.num_partitions)
        count_vector -= parameters.num_source_rows / parameters.num_partitions

        # Add multicasted values
        count_vector += assignment.batch_union(batches)

        # Compute unicasts by summing the negative elements
        load_1_unicasts = abs((count_vector * (count_vector < 0)).sum())

        # Strategy 2 multicasts for subsets of size sq-1 to muq
        load_2_multicasts = parameters.B_j(parameters.sq - 1) / (parameters.sq - 1)
        load_2_multicasts *= parameters.num_source_rows * parameters.num_outputs
        load_2_multicasts += load_1_multicasts

        load_2_batches = set()
        for multicast_subset in itertools.combinations(servers_without_q, parameters.sq - 1):
            load_2_batches.update(set.intersection(*[assignment.labels[x]
                                                     for x in multicast_subset]))

        # Add the new unique batches
        count_vector += assignment.batch_union(load_2_batches - batches)

        # Compute unicasts
        load_2_unicasts = abs((count_vector * (count_vector < 0)).sum())

        # Store result. Unicasts are computed for 1 out of q servers
        # so need to be multiplied by q.
        results.append({'unicasts_strat_1': load_1_unicasts * parameters.q,
                        'unicasts_strat_2': load_2_unicasts * parameters.q,
                        'multicasts_strat_1': load_1_multicasts,
                        'multicasts_strat_2': load_2_multicasts})

    return pd.DataFrame(results)
