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

'''This module provides functionality for analytically evaluating the
performance of an assignment. This is only possible in some cases.

'''

import pandas as pd
import math
import model
import assignments

def uncoded_performance(par, num_samples=1):
    '''Compute load and delay for an uncoded scheme.

    Args:
    par: A system parameters object.

    num_samples: Unused
    '''
    load = par.num_source_rows * (1 - par.server_storage)
    load *= par.num_outputs / par.q
    load /= par.num_source_rows
    return pd.DataFrame({'delay': [par.computational_delay(q=par.num_servers)],
                         'load': [load]})

def mds_performance(par, num_samples=1):
    '''Compute load/delay for an MDS code.

    Args:
    par: A system parameters object.

    num_samples: Unused

    '''
    assert isinstance(par, model.SystemParameters)
    load = par.unpartitioned_load()
    delay = par.computational_delay()
    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    batches_to_wait_for = par.q * batches_per_server
    return pd.DataFrame({'load': [load], 'delay': [delay],
                         'batches': [batches_to_wait_for],
                         'servers': [par.q]})

def lt_performance(par, num_samples=1, overhead=1):
    '''Estimate LDPC performance.

    Args:
    par: A system parameters object.

    overhead: The overhead is the percentage increase in number of
    servers required to decode. 1.10 means that an additional 10%
    servers is required.

    Returns: The estimated performance.

    '''
    assert isinstance(par, model.SystemParameters)
    load = par.unpartitioned_load()
    servers_to_wait_for = math.ceil(par.q * overhead)
    delay = par.computational_delay(q=servers_to_wait_for)
    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    batches_to_wait_for = servers_to_wait_for * batches_per_server
    return pd.DataFrame({'batches': [batches_to_wait_for], 'delay': [delay],
                         'servers': [servers_to_wait_for], 'load': [load]})

def average_heuristic(par, num_samples=1):
    '''Evaluate the performance of the heuristic assignment.'''

    assert isinstance(par, model.SystemParameters)
    result = computational_delay_heuristic_average(par)
    return result

def block_diagonal_upper_bound(parameters, assignment, num_samples=0):
    '''Compute an upper bound of the computational delay for a
    block-diagonal code with given assignment.

    Args:

    parameters: Parameters object.

    assignment: Assignment object.

    num_samples: Unused

    Returns: A dict with upper bounded computational delay.

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(assignment, assignments.Assignment)

    value_counts = [[0] * parameters.num_servers for _ in range(parameters.num_partitions)]

    # Select partition to compute upper bound for
    partition = parameters.num_partitions

    # Count values of the selected partition
    count_by_server = [0] * parameters.num_servers
    row_index = 0
    for row in assignment.rows_iterator():
        row_counts = [assignment.gamma + col for col in row]

        for i in range(parameters.num_partitions):
            for server_index in assignment.batch_labels[row_index]:
                value_counts[i][server_index] += row_counts[i]

                # count_by_server[server_index] += num_values

        row_index += 1

    # Select worst possible servers
    muq = round(parameters.server_storage * parameters.q)
    best_server_count = math.inf
    worst_server_count = 0
    for partition_index in range(parameters.num_partitions):
        reversed(sorted(value_counts[partition_index]))
        num_servers = 0
        total_values = 0

        # Collect values until decodeable
        for num_values in value_counts[partition_index]:
            num_servers += 1
            total_values += num_values
            if total_values >= parameters.rows_per_partition * muq:
                break

        if num_servers < best_server_count:
            best_server_count = num_servers

        if num_servers > worst_server_count:
            worst_server_count = num_servers


    # print(best_server_count, worst_server_count)

    # Return results
    num_servers = worst_server_count
    delay = parameters.computational_delay(q=num_servers)
    coded_rows_per_server = round(parameters.num_source_rows * parameters.server_storage)
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    batches_waited_for = num_servers * batches_per_server
    return pd.DataFrame({'batches': [batches_waited_for], 'delay': [delay],
                         'servers': [num_servers]})

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
    rows_per_element = par.num_coded_rows / (par.num_partitions * par.num_batches)
    batches_waited_for = math.ceil(par.num_source_rows / (par.num_partitions * rows_per_element) * muq)

    # Assign the remaining rows in a block-diagonal fashion
    remaining_rows_per_batch = round(par.rows_per_batch - rows_per_element * par.num_partitions)

    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0
    servers_waited_for = batches_waited_for / batches_per_server
    delay = par.computational_delay(q=int(servers_waited_for))
    return pd.DataFrame({'servers': [servers_waited_for],
                         'batches': [batches_waited_for],
                         'delay': [delay]})
