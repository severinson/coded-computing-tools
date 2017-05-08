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

'''This module contains code relating to the system model presented
in the paper by Li et al, as well as the integer programming
formulation presented by us. This module is used primarily by the
various solvers.

This module consists of the SystemParameters object, which is used to
pass around parameters associated with the distributed computing
system under simulation.

'''

import math
import functools
from scipy.misc import comb as nchoosek
import complexity

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
    def fixed_complexity_parameters_row(cls, rows_per_server=None,
                                        rows_per_partition=None,
                                        num_servers=None,
                                        code_rate=None, muq=None,
                                        num_columns=None):
        '''Attempt to find a set of servers with a fixed number of rows per
        server and rows per partition.

        '''
        raise NotImplementedError('Not implemented yet.')
        assert isinstance(rows_per_server, int) and rows_per_server > 0
        assert isinstance(rows_per_partition, int) and rows_per_partition > 0
        assert isinstance(num_servers, int) and num_servers > 0
        assert isinstance(code_rate, int) or isinstance(code_rate, float)
        assert isinstance(muq, int) and muq > 0
        assert isinstance(num_columns, int) and num_columns > 0
        num_source_rows = 1
        while True:
            server_storage = rows_per_server
            q = num_servers * code_rate


    @classmethod
    def fixed_complexity_parameters(cls, rows_per_server=None,
                                    rows_per_partition=None,
                                    min_num_servers=None,
                                    code_rate=None, muq=None,
                                    num_columns=None):
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

        return cls(rows_per_batch=rows_per_batch,
                   num_servers=num_servers, q=q, num_outputs=q,
                   server_storage=server_storage,
                   num_partitions=num_partitions,
                   num_columns=num_columns)

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

    def __repr__(self):
        return str(self.asdict())

    # def __str__(self):
    #     string = '------------------------------\n'
    #     string += 'System Parameters:\n'
    #     string += '------------------------------\n'
    #     string += 'Rows per batch:  ' + str(self.rows_per_batch) + '\n'
    #     string += 'Servers: ' + str(self.num_servers) + '\n'
    #     string += 'Wait for: ' + str(self.q) + '\n'
    #     string += 'Input/output vectors: ' + str(self.num_outputs) + '\n'
    #     string += 'Storage: ' + str(self.server_storage) + '\n'
    #     string += 'Batches: ' + str(self.num_batches) + '\n'
    #     string += 'muq: ' + self.muq + '\n'
    #     string += 'Code rate: ' + str(self.q / self.num_servers) + '\n'
    #     string += 'Source rows: ' + str(self.num_source_rows) + '\n'
    #     string += 'Coded rows: ' + str(self.num_coded_rows) + '\n'
    #     string += 'Columns: ' + str(self.num_columns) + '\n'
    #     string += 'Partitions: ' + str(self.num_partitions) + '\n'
    #     string += 'Rows per partition: ' + str(self.rows_per_partition) + '\n'
    #     string += 'Multicast messages: ' + str(self.num_multicasts()) + '\n'
    #     string += 'Unpartitioned load: ' + str(self.unpartitioned_load()) + '\n'
    #     string += '------------------------------'
    #     return string

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
    def multicast_set_size_1(self):
        '''Compute the size of the smallest multicast set using strategy 1.
        Denoted by s_q in the paper.

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
            if cumsum > 1 - self.server_storage:
                return max(min(set_size + 1, self.muq), 2)
            set_size -= 1

        return 2

    def multicast_set_size_2(self):
        '''Compute the size of the smallest multicast set using strategy 2.
        Denoted by s_q - 1 in the paper.

        Raises:

        ModelError: Raised if strategy 2 isn't available for these
        parameters.

        '''
        if self.multicast_set_size_1() < 3:
            raise ModelError('Shuffling strategy 2 requires multicast_set_size_1 to be at least 3.')
        return self.multicast_set_size_1() - 1

    @functools.lru_cache(maxsize=8)
    def multicast_load(self):
        '''Compute the multicast load for strategy 1 and 2.

        Returns: A tuple (multicast_load_1, multicast_load_2)

        '''
        load_1 = 0
        load_2 = 0

        # Compute load 1 and 2 (leaving out the last term of load 2).
        try:
            for j in range(self.multicast_set_size_1(), self.muq + 1):
                alpha = self.alphaj(j)
                load_1 += alpha / j
                load_2 += alpha / j
        except ModelError:
            pass

        # Add last term of load 2, or set to infinity if unavailable.
        try:
            load_2 += self.alphaj(self.multicast_set_size_2())
        except ModelError:
            load_2 = math.inf

        # Scale by outputs and return.
        load_1 *= self.num_outputs
        load_2 *= self.num_outputs
        return load_1, load_2

    @functools.lru_cache(maxsize=8)
    def unpartitioned_load(self, strategy='best', multicast_cost=1):
        '''Compute the communication load of the unpartitioned scheme.

        Args:

        strategy: Can be '1', '2', or 'best'. 'best' selects the
        strategy minimizing the load.

        multicast_cost: The cost of multicasting relative to
        unicasting.

        Returns: Total number of messages per source row.

        Raises:

        ModelError: If strategy 2 is selected when not available.

        '''
        assert strategy == 'best' or strategy == '1' or strategy == '2'
        assert isinstance(multicast_cost, int) or isinstance(multicast_cost, float)
        assert 0 < multicast_cost < math.inf

        # Unicasting load
        load_1 = 1 - self.server_storage
        load_2 = 0
        try:
            for j in range(self.multicast_set_size_1(), self.muq + 1):
                alpha = self.alphaj(j)
                load_1 -= alpha
        except ModelError:
            pass
        load_1 *= self.num_outputs

        # Multicasting load
        multicast_load_1, multicast_load_2 = self.multicast_load()
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

    @functools.lru_cache(maxsize=8)
    def computational_delay(self, q=None):
        '''Return the delay incurred in the map phase.

        Calculates the computational delay assuming a shifted
        exponential distribution.

        Args:
        q: The number of servers to wait for. If q is None, the value in self is used.

        Returns: The normalized computational delay. Multiply the
        result by
        complexity.matrix_vector_complexity(self.server_storage *
        self.num_source_rows, # self.num_columns) for the absolute
        result.

        '''
        assert q is None or isinstance(q, int)
        if q is None:
            q = self.q

        # Return infinity if waiting for more than num_servers servers
        if q > self.num_servers:
            return math.inf

        delay = 1
        for j in range(self.num_servers - q + 1, self.num_servers):
            delay += 1 / j

        # Scale by number of output vectors
        delay *= self.num_outputs
        return delay

    def computational_delay_ldpc(self, q=None, overhead=1.10):
        '''Estimate the theoretical computational delay when using an LDPC
        code.

        Args:

        q: The number of servers to wait for. If q is None, the value
        in self is used.

        overhead: The overhead is the percentage increase in number of
        servers required to decode. 1.10 means that an additional 10%
        servers is required.

        Returns: The estimated computational delay.

        '''
        assert isinstance(q, int) or q is None
        assert isinstance(overhead, float) or isinstance(overhead, int)
        if q is not None:
            assert q >= 1

        if q is None:
            q = self.q

        servers_to_wait_for = math.ceil(q * overhead)

        # Return infinity if waiting for more than num_servers servers
        if servers_to_wait_for > self.num_servers:
            return math.inf

        delay = 1
        for j in range(self.num_servers - servers_to_wait_for + 1, self.num_servers):
            delay += 1 / j

        delay *= self.server_storage * self.num_outputs
        return delay

    @functools.lru_cache(maxsize=8)
    def reduce_delay(self, num_partitions=None):
        '''Return the delay incurred in the reduce step.

        Calculates the reduce delay assuming a shifted exponential
        distribution.

        Args:
        num_partitions: The number of partitions. If None, the value in self is used.

        Returns:
        The reduce delay.

        '''
        assert num_partitions is None or isinstance(num_partitions, int)
        if num_partitions is None:
            num_partitions = self.num_partitions

        delay = 1
        for j in range(1, self.q):
            delay += 1 / j

        # Scale by decoding complexity
        delay *= complexity.block_diagonal_decoding_complexity(self.num_coded_rows,
                                                               1,
                                                               1 - self.q / self.num_servers,
                                                               num_partitions)
        delay *= self.num_outputs / self.q
        delay /= self.num_source_rows
        return delay

    def reduce_delay_ldpc_peeling(self):
        '''Return the delay incurred in the reduce step when using an LDPC
        code and a peeling decoder..

        Calculates the reduce delay assuming a shifted exponential
        distribution.

        Returns:
        The reduce delay.

        '''
        delay = 1
        for j in range(1, self.q):
            delay += 1 / j

        code_rate = self.q / self.num_servers
        delay *= complexity.peeling_decoding_complexity(self.num_coded_rows,
                                                        code_rate,
                                                        1 - self.q / self.num_servers)
        delay *= self.num_outputs / self.q
        delay /= self.num_source_rows# * self.num_outputs
        return delay
