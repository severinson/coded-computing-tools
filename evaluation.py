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

""" This module contains code relating to performance evaluation. """

import math
import fractions
import random
import itertools as it
from scipy.misc import comb as nchoosek
import numpy as np
import model

def communication_load(par, assignment_matrix, labels, Q=None):
    """ Calculate the communication load of an assignment.

    Count the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment exhaustively.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    Q: Leave at None to evaluate all subsets Q, or set to a tuple of servers to
    count unicasts only for some specific Q.

    Returns:
    Tuple with the first element the average number of unicasts required and
    second element the worst case number of unicasts.
    """

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        finished_servers_subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        finished_servers_subsets = [Q]
        num_subsets = 1

    for finished_servers in finished_servers_subsets:
        set_score = 0

        # Count over all server perspectives
        for server in finished_servers:
            # Create the set of all symbols stored at k or sent to k
            # via multicast.
            rows = set()

            for batch in labels[server]:
                rows.add(batch)

            for j in range(par.sq, int(par.server_storage*par.q) + 1):
                for multicast_subset in it.combinations([x for x in finished_servers if x != server], j):
                    rows = rows | set.intersection(*[labels[x] for x in multicast_subset])

            selector_vector = np.zeros(par.num_batches)
            for row in rows:
                selector_vector[row] = 1

            count_vector = np.dot(selector_vector, assignment_matrix)
            count_vector -= par.num_source_rows / par.num_partitions
            score = model.remaining_unicasts(count_vector)

            total_score = total_score + score
            set_score = set_score + score

        if set_score > worst_score:
            worst_score = set_score

    average_score = total_score / num_subsets
    return average_score, worst_score

def computational_delay(par, assignment_matrix, labels, Q=None):
    """ Calculate the computational delay of an assignment.

    Calculate the number of servers required to decode all partitions for an
    assignment exhaustively.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.

    Returns:
    Number of servers required averaged over all subsets.
    """

    raise NotImplementedError('There is a bug in this function. When computing the delay, it will only add at most one additional server over q. Use the sampled version instead.')

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        finished_servers_subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        finished_servers_subsets = [Q]
        num_subsets = 1

    for finished_servers in finished_servers_subsets:
        set_score = 0

        # Count the total number of symbols per partition
        rows_by_partition = np.zeros(par.num_partitions)

        # Keep track of the batches we've added
        batches = set()

        # Add the batches from the first q servers
        for server in finished_servers:
            for batch in labels[server]:
                batches.add(batch)

        for batch in batches:
            rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])
        set_score = par.q

        # TODO: Add more servers until all partitions can be decoded
        if not enough_symbols(par, rows_by_partition):
            set_score += 1

        delay = par.computational_delay(q=set_score)

        total_score += delay
        if delay > worst_score:
            worst_score = delay

    average_score = total_score / num_subsets
    return average_score, worst_score

def enough_symbols(par, rows_by_partition):
    """" Return True if there are enough symbols to decode all partitions.

    Args:
    par: System parameters object
    rows_by_partition: A numpy array of row counts by partition.

    Returns:
    True if there are enough symbols to decode all partitions. Otherwise false.
    """

    assert len(rows_by_partition) == par.num_partitions, \
        'The input array must have length equal to the number of partitions.'

    for num_rows in rows_by_partition:
        if num_rows < par.rows_per_partition:
            return False

    return True

def computational_delay_sampled(par, assignment_matrix, labels, num_samples):
    """ Estimate the computational delay of an assignment.

    Estimate the number of servers required to decode all partitions for an
    assignment through Monte Carlo simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    num_runs: Number of runs

    Returns:
    Number of servers required averaged over all n runs.
    """
    result = {'batches': 0, 'servers': 0, 'delay': 0}

    coded_rows_per_server = par.num_source_rows * par.server_storage
    assert coded_rows_per_server % 1 == 0
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0

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
            for batch in labels[server]:
                batches_added.add(batch)

        for batch in batches_added:
            rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])

        # Add batches from more servers until there are enough rows to
        # decode all partitions.
        for server in finished_servers[par.q:]:
            if enough_symbols(par, rows_by_partition):
                break

            # Update the score
            servers_waited_for += 1

            # Add the rows from the batches not already counted
            for batch in labels[server]:
                if enough_symbols(par, rows_by_partition):
                    break

                batches_waited_for += 1
                if batch in batches_added:
                    continue

                # Add the rows in the batch
                rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])

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

def communication_load_sampled(par, assignment_matrix, labels, num_samples):
    """ Estimate the communication load of an assignment.

    Estimate the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment through Monte Carlo
    simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the
    rows of the assignment_matrix stored at server i.
    num_runs: Number of runs

    Returns:
    Number of unicasts required averaged over all n runs.
    """

    total_load = 0
    server_list = list(range(par.num_servers))

    for _ in range(num_samples):

        # Generate a random set of finished servers and select one
        finished_servers = random.sample(server_list, par.q)
        finished_server = random.sample(finished_servers, 1)[0]

        # Sum the corresponding rows of the assignment matrix
        batches = set()
        for batch in labels[finished_server]:
            batches.add(batch)

        # Add the rows sent in the shuffle phase
        for j in range(par.sq, int(par.server_storage*par.q) + 1):
            for multicast_subset in it.combinations([x for x in finished_servers
                                                     if x != finished_server], j):
                batches = batches | set.intersection(*[labels[x] for x in multicast_subset])

        count_vector = np.zeros(par.num_partitions) - par.num_source_rows / par.num_partitions
        for batch in batches:
            count_vector = np.add(count_vector, assignment_matrix[batch])

        # Calculate the score
        score = model.remaining_unicasts(count_vector)
        total_load += score

    average_load = total_load * par.q / num_samples
    return {'load': average_load}

def objective_function_sampled(par, assignment_matrix, labels, num_samples=1000):
    """Return the estimated communication load and computational delay.

    Estimate the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment through Monte Carlo
    simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    n: Number of runs

    Returns: Dict with entries for estimated number of unicasts, and
    estimated number of servers required to wait for.

    """

    result = dict()
    result.update(computational_delay_sampled(par, assignment_matrix, labels, num_samples))
    result.update(communication_load_sampled(par, assignment_matrix, labels, num_samples))
    return result

def mds_performance(par, num_samples=1):
    '''Compute load/delay for an MDS code.

    Args:
    par: A system parameters object.

    num_samples: Unused

    '''
    load = par.unpartitioned_load()
    delay = par.computational_delay()
    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    batches_to_wait_for = par.q * batches_per_server
    return {'load': load, 'delay': delay, 'servers': par.q, 'batches': batches_to_wait_for}

def lt_performance(par, num_samples=1, overhead=1):
    '''Estimate LDPC performance.

    Args:
    par: A system parameters object.

    overhead: The overhead is the percentage increase in number of
    servers required to decode. 1.10 means that an additional 10%
    servers is required.

    Returns: The estimated performance.

    '''
    servers_to_wait_for = math.ceil(par.q * overhead)
    delay = par.computational_delay(q=servers_to_wait_for)
    load = math.inf

    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    batches_to_wait_for = servers_to_wait_for * batches_per_server
    return {'load': load, 'batches': batches_to_wait_for,
            'servers': servers_to_wait_for, 'delay': delay}

def upper_bound_heuristic(par, num_samples=1):
    '''Evaluate the performance of the heuristic assignment.'''

    # Load is not inplemented yet
    result = {'load': math.inf}
    result.update(computational_delay_heuristic_upper(par))
    return result

def computational_delay_heuristic_upper(par, num_samples=1):
    '''Evaluate the computational of the heuristic assignment.

    Compute an upper bound of the computational delay when using an
    heuristic assignment.

    Args:

    par: A system parameters object.

    num_samples: Unused, but required to be conformant to other
    evaluation methods.

    Return:
    An upper bound of the computational delay.

    '''

    muq = int(par.q * par.server_storage)

    # If there are more coded rows than matrix elements, add that
    # number of rows to each element.
    rows_per_element = math.floor(par.num_coded_rows / (par.num_partitions * par.num_batches))

    # Assign the remaining rows in a block-diagonal fashion
    remaining_rows_per_batch = round(par.rows_per_batch - rows_per_element * par.num_partitions)

    # Case 1 only applies for a positive number of rows per element
    if rows_per_element > 0:
        # The number of batches needed to store m/T rows from each
        # partition minus 1.
        case_1 = math.ceil(par.num_source_rows / (par.num_partitions * rows_per_element)) - 1

        # In the worst case we need to get muq of each batch.
        case_1 *= muq

        # Except for the last batch, where we can exit after receiving
        # the first repeated batch.
        case_1 += 1

    else:
        case_1 = math.inf

    # Case 2 only applies if the assignment matrix has a
    # block-diagonal structure.
    if remaining_rows_per_batch > 0:
        # The fraction of partitions covered by a block
        block_fraction = fractions.Fraction(remaining_rows_per_batch, par.num_partitions)

        # The block cycle length
        cycle_length = block_fraction.numerator

        # The number of block cycles
        block_cycles = par.num_batches / block_fraction.denominator

        # Misses before first wrap
        misses_per_cycle = math.ceil(1 / block_fraction) - 1

        # Misses for all subsequent wraps
        misses_per_cycle += (block_fraction.numerator - 1) * (math.ceil(1 / block_fraction) - 2)

        missed_batches = misses_per_cycle * block_cycles

        print('BF:', block_fraction, 'CL:', cycle_length, 'BC:', block_cycles, 'MPC:', misses_per_cycle, 'MB:', missed_batches, 'B:', par.num_batches)

        # Number of batches before all but the last block is exhausted
        case_2 = missed_batches

        # Number of batches needed from the last block minus 1.
        # remaining_rows_needed = max(par.num_source_rows / par.num_partitions - rows_per_element * missed_batches, 0)
        remaining_rows_needed = par.num_source_rows / par.num_partitions - rows_per_element * missed_batches
        case_2 += remaining_rows_needed / (rows_per_element + 1)
        case_2 -= 1
        case_2 *= muq
        case_2 += 1

    else:
        case_2 = math.inf

    # batches_waited_for = min(case_1, case_2, par.num_batches * muq)
    batches_waited_for = min(case_1, case_2)

    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    servers_waited_for = math.ceil(batches_waited_for / batches_per_server)

    # servers_waited_for = min(servers_waited_for, par.num_servers)
    delay = par.computational_delay(q=servers_waited_for)
    print('Min', batches_waited_for, 'Case 1', case_1, 'Case 2', case_2, 'Server:', servers_waited_for)
    print()
    return {'batches': batches_waited_for, 'servers': servers_waited_for, 'delay': delay}

def average_heuristic(par, num_samples=1):
    '''Evaluate the performance of the heuristic assignment.'''

    # Load is not inplemented yet
    result = {'load': math.inf}
    result.update(computational_delay_heuristic_average(par))
    return result

def computational_delay_heuristic_average(par, num_samples=1):
    '''Evaluate the computational of the heuristic assignment.

    Compute a lower bound of the computational delay when using an
    heuristic assignment.

    Args:

    par: A system parameters object.

    num_samples: Unused, but required to be conformant to other
    evaluation methods.

    Returns:
    A lower bound of the computational delay.

    '''

    muq = int(par.q * par.server_storage)

    # If there are more coded rows than matrix elements, add that
    # number of rows to each element.
    # rows_per_element = math.floor(par.num_coded_rows / (par.num_partitions * par.num_batches))
    rows_per_element = par.num_coded_rows / (par.num_partitions * par.num_batches)
    batches_waited_for = math.ceil(par.num_source_rows / (par.num_partitions * rows_per_element) * muq)

    # Assign the remaining rows in a block-diagonal fashion
    remaining_rows_per_batch = round(par.rows_per_batch - rows_per_element * par.num_partitions)

    rows_per_batch = rows_per_element + remaining_rows_per_batch / par.num_partitions
    # batches_waited_for = math.ceil(par.num_source_rows / (par.num_partitions * rows_per_batch) * muq)
    #batches_waited_for *= muq
    #batches_waited_for += 1

    coded_rows_per_server = par.num_source_rows * par.server_storage
    assert coded_rows_per_server % 1 == 0
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0
    servers_waited_for = batches_waited_for / batches_per_server

    delay = par.computational_delay(q=int(servers_waited_for))
    print('SWF:', servers_waited_for, 'RPE:', rows_per_element, 'RRPB:', remaining_rows_per_batch, 'RPB:', rows_per_batch)
    print()

    return {'servers': servers_waited_for, 'batches': batches_waited_for, 'delay': delay}

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

    Returns:
    A dict with the simulated results.
    '''

    result = dict()
    result.update(computational_delay_unsupervised_average(par, num_samples=num_samples))
    result.update(communication_load_unsupervised(par, num_samples=num_samples))
    return result

def communication_load_unsupervised(par, num_samples=1000):

    '''Placeholder function for a simplified communication load analysis.

    '''
    return {'load': math.inf}

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
    cum_delay = 0
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

    # Round to avoid numerical problems as mu can be very small.
    # The product is guaranteed to be integer.
    rows_per_server = round(par.num_source_rows * par.server_storage)

    coded_rows_per_partition = par.num_servers / par.q * par.rows_per_partition
    assert coded_rows_per_partition % 1 == 0
    coded_rows_per_partition = int(coded_rows_per_partition)

    # Setup sets for counting rows for each partition
    partitions = dict()
    for partition_index in range(par.num_partitions):
        partitions[partition_index] = set()

    # Add the rows stored at the first q servers to finish
    batches_waited_for = 0
    while len(partitions) > 0:
        batches_waited_for += 1

        # Generate a random set of partition-symbol index pairs
        partition_indices = [random.choice(range(par.num_partitions)) for _ in range(par.rows_per_batch)]
        symbol_indices = [random.choice(range(coded_rows_per_partition)) for _ in range(par.rows_per_batch)]
        for partition_index, symbol_index in zip(partition_indices, symbol_indices):

            if partition_index not in partitions:
                continue

            # Add the symbols to their partitions
            partitions[partition_index].add(symbol_index)

            # Remove the partition when we've collected enough rows
            if len(partitions[partition_index]) >= par.rows_per_partition:
                del partitions[partition_index]


    coded_rows_per_server = par.num_source_rows * par.server_storage
    assert coded_rows_per_server % 1 == 0
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0
    servers_waited_for = math.ceil(batches_waited_for / batches_per_server)
    delay = par.computational_delay(q=servers_waited_for)

    return {'batches': batches_waited_for, 'servers': servers_waited_for, 'delay': delay}
