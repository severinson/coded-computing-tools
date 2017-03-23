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

'''This module provides functionality for sampling the performance of
an assignment. Specifically, the performance is evaluated through
Monte Carlo simulations by randomizing the order in which servers
finish.

'''

import math
import random
import itertools
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
    result.update(computational_delay_sampled(par, assignment, num_samples))
    result.update(communication_load_sampled(par, assignment, num_samples))
    return result

def enough_symbols(par, rows_by_partition):
    '''' Return True if there are enough symbols to decode all partitions.

    Args:
    par: System parameters object

    rows_by_partition: A numpy array of row counts by partition.

    Returns: True if there are enough symbols to decode all
    partitions. Otherwise false.

    '''
    assert isinstance(par, model.SystemParameters)
    assert len(rows_by_partition) == par.num_partitions, \
        'The input array must have length equal to the number of partitions.'

    for num_rows in rows_by_partition:
        if num_rows < par.rows_per_partition:
            return False

    return True

def computational_delay_sampled(par, assignment, num_samples):
    '''Sample the computational delay of an assignment.

    Estimate the number of servers required to decode all partitions
    for an assignment through Monte Carlo simulations.

    Args:
    par: System parameters

    assignment_matrix: An Assignment object.

    num_runs: Number of runs

    Returns: Average Number of servers required.

    '''
    assert isinstance(par, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(num_samples, int)
    result = {'batches': 0, 'servers': 0, 'delay': 0}

    coded_rows_per_server = par.num_source_rows * par.server_storage
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    for _ in range(num_samples):
        servers_waited_for = par.q
        batches_waited_for = par.q * batches_per_server

        # Generate a random sequence of servers
        finished_servers = random.sample(range(par.num_servers), par.num_servers)

        # Count the total number of symbols per partition
        rows_by_partition = np.zeros(par.num_partitions)

        # Keep track of the batches we've added
        batches_added = set()

        # Add the batches from the first q servers
        for server in finished_servers[0:par.q]:
            for batch in assignment.labels[server]:
                batches_added.add(batch)

        rows_by_partition = np.add(rows_by_partition, assignment.batch_union(batches_added))

        # Add batches from more servers until there are enough rows to
        # decode all partitions.
        for server in finished_servers[par.q:]:
            if enough_symbols(par, rows_by_partition):
                break

            # Update the score
            servers_waited_for += 1

            # Add the rows from the batches not already counted
            for batch in assignment.labels[server]:
                if enough_symbols(par, rows_by_partition):
                    break

                batches_waited_for += 1
                if batch in batches_added:
                    continue

                # Add the rows in the batch
                rows_by_partition = np.add(rows_by_partition, assignment.batch_union({batch}))

                # Keep track of which batches we've added
                batches_added.add(batch)

        delay = par.computational_delay(q=servers_waited_for)
        result['batches'] += batches_waited_for
        result['servers'] += servers_waited_for
        result['delay'] += delay

    result['batches'] /= num_samples
    result['servers'] /= num_samples
    result['delay'] /= num_samples
    return result

def communication_load_sampled(par, assignment, num_samples):
    '''Sample the communication load of an assignment.

    Estimate the number of unicasts required for all servers to hold
    enough symbols to deocde all partitions for some assignment
    through Monte Carlo simulations.

    Args:
    par: System parameters

    assignment_matrix: An Assignment object.

    num_runs: Number of runs

    Returns: Average number of unicasts required.

    '''
    assert isinstance(par, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)
    assert isinstance(num_samples, int)

    total_load = 0
    server_list = list(range(par.num_servers))

    for _ in range(num_samples):

        # Generate a random set of finished servers and select one
        finished_servers = random.sample(server_list, par.q)
        finished_server = random.sample(finished_servers, 1)[0]

        # Sum the corresponding rows of the assignment matrix
        batches = set()
        for batch in assignment.labels[finished_server]:
            batches.add(batch)

        # Add the rows sent in the shuffle phase
        for j in range(par.sq, int(par.server_storage*par.q) + 1):
            for multicast_subset in itertools.combinations([x for x in finished_servers
                                                            if x != finished_server], j):
                batches = batches | set.intersection(*[assignment.labels[x] for x in multicast_subset])

        count_vector = np.zeros(par.num_partitions) - par.num_source_rows / par.num_partitions
        count_multicasts = assignment.batch_union(batches)
        count_vector = np.add(count_vector, count_multicasts)

        # Calculate the score
        score = model.remaining_unicasts(count_vector)
        total_load += score

    average_load = total_load * par.q / num_samples
    return {'load': average_load}

def eval_unsupervised(par, num_samples=1000):
    '''Run a simplified performance analysis for a set of system
    parameters.

    This function simulates the performance based only on the system
    parameters rather than generating an assignment and sampling its
    performance. Rows are assumed to be allocated to servers randomly,
    and there is no limit to the number of times a row may be
    allocated to a server.

    Args:
    par: A system parameters object.

    Returns: A dict with the simulated results.

    '''

    result = dict()
    result['load'] = math.inf
    result.update(computational_delay_unsupervised_average(par, num_samples=num_samples))
    return result

def computational_delay_unsupervised_average(par, num_samples=1000):
    '''Simulate the required number of servers of a system under some
    simplifying assumptions.

    Rows are assumed to be allocated to servers randomly, and there is
    no limit to the number of times a row may be allocated to a
    server.

    Args:n
    par: A system parameters object.
    num_samples: Number of runs to average the result over.

    Returns:
    The number of servers required to decode averaged over num_samples runs.

    '''
    cum_result = {'batches': 0, 'servers': 0, 'delay': 0}
    for _ in range(num_samples):
        result = computational_delay_unsupervised(par)
        cum_result['batches'] += result['batches']
        cum_result['servers'] += result['servers']
        cum_result['delay'] += result['delay']

    cum_result['batches'] /= num_samples
    cum_result['servers'] /= num_samples
    cum_result['delay'] /= num_samples
    return cum_result

def computational_delay_unsupervised(par):
    '''Simulate the required number of servers of a system under some
    simplifying assumptions.

    This functions runs the simulation once. You probably want to call
    computational_delay_simplified() for an averaged result. Rows are
    assumed to be allocated to servers randomly, and there is no limit
    to the number of times a row may be allocated to a server.

    Args:
    par: A system parameters object.

    Returns:
    The number of servers required to decode.

    '''

    coded_rows_per_partition = par.num_servers / par.q * par.rows_per_partition
    assert coded_rows_per_partition % 1 == 0
    coded_rows_per_partition = int(coded_rows_per_partition)

    # Setup sets for counting rows for each partition
    partitions = dict()
    for partition_index in range(par.num_partitions):
        partitions[partition_index] = set()

    # Setup a set with the indices of all incomplete partitions
    incomplete_partitions = {index for index in range(par.num_partitions)}

    # Add the rows stored at the first q servers to finish
    batches_waited_for = 0
    while incomplete_partitions:
        batches_waited_for += 1

        # Generate a random set of partition-symbol index pairs
        partition_indices = [random.choice(range(par.num_partitions))
                             for _ in range(par.rows_per_batch)]

        symbol_indices = [random.choice(range(coded_rows_per_partition))
                          for _ in range(par.rows_per_batch)]

        for partition_index, symbol_index in zip(partition_indices, symbol_indices):

            # Add the symbols to their partitions
            partitions[partition_index].add(symbol_index)

            # Mark as complete if we've collected enough rows
            if len(partitions[partition_index]) >= par.rows_per_partition:
                del incomplete_partitions[partition_index]

    coded_rows_per_server = par.num_source_rows * par.server_storage
    assert coded_rows_per_server % 1 == 0
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0
    servers_waited_for = math.ceil(batches_waited_for / batches_per_server)
    delay = par.computational_delay(q=servers_waited_for)

    return {'batches': batches_waited_for,
            'servers': servers_waited_for, 'delay': delay}
