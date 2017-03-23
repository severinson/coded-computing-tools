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

import math
import model

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
    return {'load': load, 'delay': delay, 'servers': par.q,
            'batches': batches_to_wait_for}

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
    servers_to_wait_for = math.ceil(par.q * overhead)
    delay = par.computational_delay(q=servers_to_wait_for)
    load = math.inf

    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    batches_to_wait_for = servers_to_wait_for * batches_per_server
    return {'load': load, 'batches': batches_to_wait_for,
            'servers': servers_to_wait_for, 'delay': delay}

def average_heuristic(par, num_samples=1):
    '''Evaluate the performance of the heuristic assignment.'''

    assert isinstance(par, model.SystemParameters)

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
    rows_per_element = par.num_coded_rows / (par.num_partitions * par.num_batches)
    batches_waited_for = math.ceil(par.num_source_rows / (par.num_partitions * rows_per_element) * muq)

    # Assign the remaining rows in a block-diagonal fashion
    remaining_rows_per_batch = round(par.rows_per_batch - rows_per_element * par.num_partitions)

    coded_rows_per_server = round(par.num_source_rows * par.server_storage)
    batches_per_server = coded_rows_per_server / par.rows_per_batch
    assert batches_per_server % 1 == 0
    servers_waited_for = batches_waited_for / batches_per_server
    delay = par.computational_delay(q=int(servers_waited_for))
    return {'servers': servers_waited_for, 'batches': batches_waited_for, 'delay': delay}
