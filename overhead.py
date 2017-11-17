############################################################################
# Copyright 2017 Albin Severinson                                          #
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

'''Evaluate the performance of a code with a known overhead. Specifically,
compute the average performance impact of requiring any set of m(1 + eps)
unique rows to decode.

The computation is performed exhaustively when possible and via Monte Carlo
simulations otherwise.

'''

import math
import random
import itertools
import numpy as np
import pandas as pd
from scipy.misc import comb as nchoosek
import model
import numtools

from functools import lru_cache

@lru_cache()
def performance_from_overhead(parameters=None, overhead=1, num_samples=1000):
    '''compute the average performance at some fixed overhead.

    args:

    parameters: system parameters

    overhead: value such that any overhead * m unique rows is sufficient to
    decode on average.

    '''

    # check all possible completion orders or num_samples randomly selected
    # orders, whichever is smaller.
    exhaustive_samples = nchoosek(parameters.num_servers, parameters.q)
    if exhaustive_samples > num_samples:
        completion_orders = random_completion_orders(parameters, num_samples)
    else:
        exhaustive_samples *= math.factorial(parameters.num_servers - parameters.q)
        if exhaustive_samples <= num_samples:
            completion_orders = exhaustive_completion_orders(parameters)
        else:
            completion_orders = random_completion_orders(parameters, num_samples)

    results = list()
    for order in completion_orders:
        result = dict()
        result.update(
            delay_from_order(parameters, order, overhead)
        )
        result.update(
            load_from_order(parameters, overhead)
        )
        results.append(result)

    return pd.DataFrame(results)

def random_completion_orders(parameters=None, num_samples=None):
    '''generate random server completion orders'''
    for _ in range(num_samples):
        yield random.sample(range(parameters.num_servers), parameters.num_servers)
    return

def exhaustive_completion_orders(parameters=None):
    '''generate all possible server completion orders'''
    servers = set(range(parameters.num_servers))

    # Order of the first q servers is irrelevant.
    for order in itertools.combinations(range(parameters.num_servers), parameters.q):
        remaining_servers = servers - set(order)

        # Evaluate all permutations of the remaining servers.
        for remaining_order in itertools.permutations(remaining_servers):
            yield list(order) + list(remaining_order)

@lru_cache()
def _batches_by_server(num_servers=None, servers_per_batch=None):
    '''compute what server stores what coded rows.

    num_servers: total number of servers.

    servers_per_batch: number of servers that each batch is stored at.

    returns: list of length num_servers with each element a set containing the
    indices of the batches stored by that server.

    '''
    assert num_servers is not None
    assert servers_per_batch is not None

    # indices of rows stored by each server
    storage = [set() for _ in range(num_servers)]

    # generate all server labels, e.g., (1,2), (1,3), (1,4), ....
    labels = itertools.combinations(
        range(num_servers),
        int(servers_per_batch),
    )

    # add batch indices to servers
    batch_index = 0
    for label in labels:
        for server_index in label:
            storage[server_index].add(batch_index)
        batch_index += 1

    # TODO: Return frozensets?
    return storage

def _batches_from_order(storage=None, servers=None):
    assert storage is not None
    assert servers is not None
    batches = set().union(*(storage[server] for server in servers))
    return batches

def _rows_from_batches(parameters=None, batches=None):
    assert isinstance(parameters, model.SystemParameters)
    assert batches is not None
    return len(batches) * parameters.rows_per_batch

def rows_from_q(parameters=None, q=None, num_samples=1000):
    '''compute the average number of unique rows stored at q servers'''
    assert isinstance(parameters, model.SystemParameters)
    if q is None:
        q = parameters.q
    storage = _batches_by_server(
        num_servers=parameters.num_servers,
        servers_per_batch=parameters.muq,
    )

    # check all possible completion orders or num_samples randomly selected
    # orders, whichever is smaller.
    exhaustive_samples = nchoosek(parameters.num_servers, parameters.q)
    if exhaustive_samples > num_samples:
        completion_orders = random_completion_orders(parameters, num_samples)
    else:
        exhaustive_samples *= math.factorial(parameters.num_servers - parameters.q)
        if exhaustive_samples <= num_samples:
            completion_orders = exhaustive_completion_orders(parameters)
        else:
            completion_orders = random_completion_orders(parameters, num_samples)

    results = list()
    for order in completion_orders:
        batches = _batches_from_order(storage=storage, servers=order[:q])
        rows = _rows_from_batches(parameters=parameters, batches=batches)
        results.append(rows)

    return np.asarray(results).mean()

def delay_from_order(parameters=None, order=None, overhead=None):
    '''compute the delay for some overhead.'''
    assert isinstance(parameters, model.SystemParameters)
    assert order is not None
    assert overhead is not None
    storage = _batches_by_server(
        num_servers=parameters.num_servers,
        servers_per_batch=parameters.muq,
    )
    required_rows = math.ceil(parameters.num_source_rows * overhead)
    permanent_x = parameters.q
    tentative_x = permanent_x
    permanent = _batches_from_order(storage, order[:parameters.q])
    tentative = set()
    def decodeable(x):
        '''return 1 if decoding is possible with x servers and 0 otherwise'''
        nonlocal permanent, tentative, permanent_x, tentative_x
        nonlocal parameters, storage, order, required_rows

        # if x is higher than tentative_x, tentative_x servers was not enough
        # and we can safely update the set of permanent batches.
        if x > tentative_x:
            permanent.update(tentative)
            permanent_x = tentative_x
            tentative = set()

        tentative = _batches_from_order(storage, order[permanent_x:x])
        batches = permanent.union(tentative)
        if _rows_from_batches(parameters, batches) >= required_rows:
            return 1
        return 0

    required_servers = numtools.numinv(
        fun=decodeable,
        target=1,
        lower=parameters.q,
        upper=parameters.num_servers,
    )
    coded_rows_per_server = parameters.num_source_rows * parameters.server_storage
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    return {'servers': required_servers, 'batches': required_servers * batches_per_server,
            'delay': parameters.computational_delay(q=required_servers)}

def load_from_order(parameters=None, overhead=None):
    '''compute the load for some overhead.'''
    assert isinstance(parameters, model.SystemParameters)
    assert overhead is not None
    return {'load': parameters.unpartitioned_load(overhead=overhead)}
