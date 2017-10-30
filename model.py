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

'''This module consists of the SystemParameters object, which is used
to pass around parameters associated with the distributed computing
system under simulation.

'''

import math
import functools
from scipy.special import comb as nchoosek
import complexity
import stats

class ModelError(Exception):
    '''Base class for exceptions raised by this module.'''

class SystemParameters(object):
    '''System parameters representation. This object is used to pass
    around parameters associated with the distributed computing system
    under simulation. Furthermore, the object contains functions used
    to compute load and delay for various cases.

    '''

    def __init__(self, rows_per_batch=None, num_servers=None, q=None, num_outputs=None,
                 server_storage=None, num_partitions=None, num_columns=None):
        '''Create a system parameters object.

        Args:

        rows_per_batch: Number of coded rows to store per batch.

        num_servers: Number of servers used in the map phase.

        q: Number of servers to wait for in the map step and the
        number of servers used in the shuffle and reduce phases.

        num_outputs: Number of input and output vectors.

        server_storage: The number of coded rows stored by each server
        in the map phase is given by server_storage*num_source_rows.

        num_partitions: Number of partitions used by the
        block-diagonal code.

        num_columns: Number of columns of the source matrix. If set to
        None the matrix is assumed to be square.

        '''

        assert isinstance(rows_per_batch, int)
        assert isinstance(num_servers, int)
        assert isinstance(q, int)
        assert isinstance(num_outputs, int)
        assert isinstance(server_storage, float) or server_storage == 1
        assert server_storage <= 1, 'Server storage must be >0 and <=1.'
        assert isinstance(num_partitions, int)
        assert isinstance(num_columns, int) or num_columns is None
        if num_columns:
            assert num_columns > 0

        self.num_servers = num_servers
        self.q = q
        self.num_outputs = num_outputs
        self.server_storage = server_storage
        self.num_partitions = num_partitions
        self.rows_per_batch = rows_per_batch

        assert num_outputs / q % 1 == 0, 'num_outputs must be divisible by the number of servers.'
        assert server_storage * q % 1 == 0, 'server_storage * q must be integer.'

        num_batches = nchoosek(num_servers, server_storage*q, exact=True)
        assert num_batches % 1 == 0, 'There must be an integer number of batches.'
        num_batches = int(num_batches)
        self.num_batches = num_batches

        num_coded_rows = rows_per_batch * num_batches
        self.num_coded_rows = num_coded_rows

        num_source_rows = q / num_servers * num_coded_rows
        assert num_source_rows % 1 == 0, 'There must be an integer number of source rows.'
        num_source_rows = int(num_source_rows)
        self.num_source_rows = num_source_rows

        # Assume matrix is square if num_columns is None.
        if num_columns:
            self.num_columns = num_columns
        else:
            self.num_columns = self.num_source_rows

        assert num_source_rows / num_partitions % 1 == 0, \
            'There must be an integer number of source rows per partition.'

        assert num_coded_rows / num_partitions % 1 == 0, \
            'There must be an integer number of coded rows per partition. num_partitions: %d' \
            % num_partitions

        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition

        # Store some additional parameters for convenience
        self.muq = round(self.server_storage * self.q)
        return

    @classmethod
    def fixed_complexity_parameters(cls, rows_per_server=None,
                                    rows_per_partition=None,
                                    min_num_servers=None,
                                    code_rate=None, muq=None,
                                    num_columns=None,
                                    num_outputs_factor=1):
        '''Attempt to find a set of servers with a fixed number of rows per
        server and rows per partition.

        '''
        num_servers = min_num_servers
        while True:
            num_servers = num_servers + 1

            q = code_rate * num_servers
            if q % 1 != 0 or q == 0:
                continue
            q = int(q)

            server_storage = muq / q
            if server_storage < 1/q:
                print('No feasible solution.')
                return None

            if server_storage > 1:
                continue

            if server_storage * q % 1 != 0:
                continue

            num_source_rows = rows_per_server / muq * q
            if num_source_rows % 1 != 0:
                continue
            num_source_rows = int(num_source_rows)

            num_coded_rows = num_source_rows / code_rate
            if num_coded_rows % 1 != 0:
                continue
            num_coded_rows = int(num_coded_rows)

            num_batches = nchoosek(num_servers, int(server_storage*q), exact=True)
            if num_batches == 0:
                continue

            rows_per_batch = num_coded_rows / num_batches
            if rows_per_batch % 1 != 0:
                continue
            rows_per_batch = int(rows_per_batch)

            num_partitions = num_source_rows / rows_per_partition
            if num_partitions % 1 != 0:
                continue
            num_partitions = int(num_partitions)
            break

        if num_columns is None:
            num_columns = num_source_rows

        num_outputs = num_outputs_factor * q
        return cls(rows_per_batch=rows_per_batch, num_servers=num_servers, q=q,
                   num_outputs=num_outputs, server_storage=server_storage,
                   num_partitions=num_partitions, num_columns=num_columns)

    def asdict(self):
        '''Convert this object to a dict.'''
        try:
            multicast_size_1 = self.multicast_set_size_1()
        except ModelError:
            multicast_size_1 = None

        try:
            multicast_size_2 = self.multicast_set_size_2()
        except ModelError:
            multicast_size_2 = None

        dct = {'rows_per_batch': self.rows_per_batch, 'servers': self.num_servers,
               'wait_for': self.q, 'inputs_outputs': self.num_outputs,
               'storage': self.server_storage, 'batches': self.num_batches,
               'muq': self.muq, 'code_rate': self.q / self.num_servers,
               'source_rows': self.num_source_rows, 'coded_rows': self.num_coded_rows,
               'columns': self.num_columns, 'partitions': self.num_partitions,
               'rows_per_partition': self.rows_per_partition,
               'multicast_size_1': multicast_size_1, 'multicast_size_2': multicast_size_2}
        return dct

    @classmethod
    def fromdct(cls, dct):
        return cls(
            rows_per_batch=int(dct['rows_per_batch']),
            num_servers=int(dct['servers']),
            q=int(dct['wait_for']),
            num_outputs=int(dct['inputs_outputs']),
            server_storage=float(dct['storage']),
            num_partitions=int(dct['partitions']),
            num_columns=int(dct['columns'])
        )

    def __repr__(self):
        return str(self.asdict())

    def identifier(self):
        '''Return a string identifier for these parameters.'''
        string = 'm_' + str(self.num_source_rows)
        string += '_K_' + str(self.num_servers)
        string += '_q_' + str(self.q)
        string += '_N_' + str(self.num_outputs)
        string += '_muq_' + str(int(self.server_storage * self.q))
        string += '_T_' + str(self.num_partitions)
        return string

    @functools.lru_cache(maxsize=128)
    def alphaj(self, j):
        '''Compute alpha_j as defined in the paper.

        Args:

        j: Integer between 0 and num_servers.

        '''
        assert isinstance(j, int) and 0 <= j <= self.num_servers
        result = nchoosek(self.q - 1, j, exact=True)
        result *= nchoosek(self.num_servers - self.q, self.muq - j, exact=True)
        result /= self.q / self.num_servers
        result /= nchoosek(self.num_servers, self.muq, exact=True)
        return result

    @functools.lru_cache(maxsize=8)
    def multicast_set_size_1(self, overhead=1):
        '''Compute the size of the smallest multicast set using strategy 1.
        Denoted by s_q in the paper.

        Args:

        overhead: Code overhead. Equal to 1 for MDS codes.

        Raises:

        ModelError: Raised if muq == 1.

        '''
        if self.muq == 1:
            raise ModelError('Multicasting requires muq to be larger than 1.')

        if self.muq == 2:
            return 2

        cumsum = 0
        set_size = self.muq
        while set_size > 1:
            cumsum += self.alphaj(set_size)
            if cumsum > overhead - self.server_storage:
                return max(min(set_size + 1, self.muq), 2)
            set_size -= 1

        return 2

    def multicast_set_size_2(self, overhead=1):
        '''Compute the size of the smallest multicast set using strategy 2.
        Denoted by s_q - 1 in the paper.

        Args:

        overhead: Code overhead. Equal to 1 for MDS codes.

        Raises:

        ModelError: Raised if strategy 2 isn't available for these
        parameters.

        '''
        if self.multicast_set_size_1(overhead=overhead) < 3:
            raise ModelError('Shuffling strategy 2 requires multicast_set_size_1 to be at least 3.')
        return self.multicast_set_size_1(overhead=overhead) - 1

    @functools.lru_cache(maxsize=128)
    def multicast_load(self, multicast_cost=None, overhead=1):
        '''Compute the multicast load for strategy 1 and 2.

        Args:

        multicast_cost: See unpartitioned_load()

        overhead: Code overhead. Equal to 1 for MDS codes.

        Returns: A tuple (multicast_load_1, multicast_load_2)

        '''
        if not multicast_cost:
            multicast_cost = lambda j: j

        load_1 = 0
        load_2 = 0

        # Compute load 1 and 2 (leaving out the last term of load 2).
        try:
            for j in range(self.multicast_set_size_1(overhead=overhead), self.muq+1):
                alpha = self.alphaj(j) / multicast_cost(j)
                load_1 += alpha
                load_2 += alpha
        except ModelError:
            pass

        # Add last term of load 2, or set to infinity if unavailable.
        try:
            j = self.multicast_set_size_2(overhead=overhead)
            load_2 += self.alphaj(j) / multicast_cost(j)
        except ModelError:
            load_2 = math.inf

        # Scale by outputs and return.
        load_1 *= self.num_outputs
        load_2 *= self.num_outputs
        return load_1, load_2

    @functools.lru_cache(maxsize=128)
    def unpartitioned_load(self, strategy='best', multicast_cost=None, overhead=1):
        '''Compute the communication load of the unpartitioned scheme.

        Args:

        strategy: Can be '1', '2', or 'best'. 'best' selects the
        strategy minimizing the load.

        multicast_cost: Function used to compute the cost of multicasting
        relative to unicasting. Must take the number of recipients j as its
        only argument and return the ratio of the cost of unicasting the same
        message to all recipients to the cost of multicasting that message to
        all recipients. For example, if the cost of multicasting is the same as
        sending a single unicasted message, j should be returned. This is the
        default.

        overhead: Code overhead. Equal to 1 for MDS codes.

        Returns: Total number of messages per source row.

        Raises:

        ModelError: If strategy 2 is selected when not available.

        '''
        assert strategy == 'best' or strategy == '1' or strategy == '2'

        # Unicasting load
        load_1 = overhead - self.server_storage
        load_2 = 0
        try:
            for j in range(self.multicast_set_size_1(overhead=overhead), self.muq+1):
                alpha = self.alphaj(j)
                load_1 -= alpha
        except ModelError:
            pass
        load_1 *= self.num_outputs

        # Multicasting load
        multicast_load_1, multicast_load_2 = self.multicast_load(
            overhead=overhead, multicast_cost=multicast_cost)
        load_1 += multicast_load_1
        load_2 += multicast_load_2

        # Return load of the selected strategy
        if strategy == 'best':
            return min(load_1, load_2)
        if strategy == '1':
            return load_1
        if strategy == '2' and load_2 == math.inf:
            raise ModelError('Strategy 2 not available.')
        elif strategy == '2':
            return load_2

    @functools.lru_cache(maxsize=128)
    def computational_delay(self, q=None, overhead=1):
        '''Return the delay incurred in the map phase.

        Calculates the computational delay assuming a shifted
        exponential distribution.

        Args:

        q: The number of servers to wait for. If q is None, the value in self is used.

        overhead: Code overhead. Equal to 1 for MDS codes.

        returns: The normalized computational delay. Multiply the result by
        complexity.matrix_vector_complexity(self.server_storage *
        self.num_columns) for the absolute result.

        '''
        assert q is None or isinstance(q, int)
        assert overhead == 1
        if q is None:
            q = self.q

        # Add the overhead factor
        # TODO: This is incorrect
        q = math.ceil(q * overhead)

        # Return infinity if waiting for more than num_servers servers
        if q > self.num_servers:
            return math.inf

        delay = stats.order_mean_shiftexp(self.num_servers, q)

        # scale by number of output vectors. we do not scale by server_storage
        # * num_source_rows here as this value varies depending on the scheme
        # (uncoded, coded MapReduce, and straggler coding). this scaling is
        # carried out in simulation.py.
        delay *= self.num_outputs

        return delay
