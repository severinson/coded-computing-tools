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

import os
import math
import random
import itertools
import numpy as np
import pandas as pd
import model
import pynumeric

from functools import lru_cache
from scipy.special import comb as nchoosek

def performance_from_overheads(
        overheads,
        parameters=None,
        design_overhead=None):
    '''sample the performance for each overhead in overheads. returns a
    dataframe with length equal to that of overheads.

    '''
    orders = random_completion_orders(parameters, len(overheads))
    results = list()
    for (order, overhead) in zip(orders, overheads):
        result = dict()
        result.update(
            delay_from_order(parameters, order, overhead)
        )
        result.update(
            load_from_order(
                parameters=parameters,
                overhead=overhead,
                design_overhead=design_overhead,
            )
        )
        results.append(result)

    return pd.DataFrame(results)

def performance_from_overhead(parameters=None, overhead=1, design_overhead=None,
                              num_samples=1000, cachedir=None):
    '''compute the average performance at some fixed overhead.

    args:

    parameters: system parameters

    overhead: value such that any overhead * m unique rows is sufficient to
    decode on average.

    design_overhead: see mode.unpartitioned_load()

    cachedir: cache simulations here.

    '''

    # returned a cached simulation if available
    if cachedir:
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        filename = os.path.join(
            cachedir,
            parameters.identifier() + '_overhead_' + str(overhead) + '.csv',
        )
        try:
            df = pd.read_csv(filename)
            if len(df) >= num_samples:
                return df[:num_samples]
        except:
            pass

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
            load_from_order(
                parameters=parameters,
                overhead=overhead,
                design_overhead=design_overhead,
            )
        )
        results.append(result)

    # cache the simulation and return
    df = pd.DataFrame(results)
    if cachedir:
        df.to_csv(filename, index=False)
    return df

def random_completion_orders(parameters=None, num_samples=None):
    '''generate random server completion orders'''
    for _ in range(num_samples):
        yield random.sample(range(parameters.num_servers), parameters.num_servers)
    return

def exhaustive_completion_orders(parameters=None):
    '''generate all possible server completion orders'''
    servers = set(range(parameters.num_servers))

    # completion order of the first q servers is irrelevant.
    for order in itertools.combinations(range(parameters.num_servers), parameters.q):
        remaining_servers = servers - set(order)

        # consider all permutations of the remaining servers.
        for remaining_order in itertools.permutations(remaining_servers):
            yield list(order) + list(remaining_order)

@lru_cache()
def _batches_by_server(num_servers=None, servers_per_batch=None):
    '''compute which server stores which batches.

    num_servers: total number of servers.

    servers_per_batch: number of servers that each batch is stored at.

    returns: list of length num_servers where each element is a set of the
    indices of the batches stored by the corresponding server.

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
    for batch_index, label in enumerate(labels):
        for server_index in label:
            storage[server_index].add(batch_index)

    # TODO: We're returning something mutable
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

    if _rows_from_batches(parameters, permanent) >= required_rows:
        coded_rows_per_server = parameters.num_source_rows * parameters.server_storage
        batches_per_server = coded_rows_per_server / parameters.rows_per_batch
        return {
            'servers': parameters.q,
            'batches': parameters.q * batches_per_server,
            'delay': parameters.computational_delay(q=parameters.q)
        }

    def decodeable(x):
        '''return 1 if decoding is possible with x servers and 0 otherwise'''
        # TODO: This includes some variables needed to cache previously
        # computed results. We currently don't do caching, however.
        nonlocal permanent, tentative, permanent_x, tentative_x
        nonlocal parameters, storage, order, required_rows
        batches = _batches_from_order(storage, order[:x])
        if _rows_from_batches(parameters, batches) >= required_rows:
            return 1
        return 0

    required_servers = pynumeric.numinv(
        fun=decodeable,
        target=1,
        lower=parameters.q,
        upper=parameters.num_servers,
    )
    coded_rows_per_server = parameters.num_source_rows * parameters.server_storage
    batches_per_server = coded_rows_per_server / parameters.rows_per_batch
    return {
        'servers': required_servers,
        'batches': required_servers * batches_per_server,
        'delay': parameters.computational_delay(q=required_servers)
    }

def load_from_order(parameters=None, overhead=None, design_overhead=None):
    '''compute the load for some overhead.'''
    assert isinstance(parameters, model.SystemParameters)
    assert overhead is not None
    load = parameters.unpartitioned_load(
        overhead=overhead,
        design_overhead=design_overhead,
    )
    return {'load': load}
