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

"""
This module contains code relating to the system model presented in the
paper by Li et al, as well as the integer programming formulation presented
by us. This module is used primarily by the various solvers.

Run the file to run the unit tests.
"""

import math
import itertools as it
import uuid
import os
import random
import tempfile
import unittest
import numpy as np
from scipy.misc import comb as nchoosek
import complexity

class SystemParameters(object):
    """ System parameters representation. This struct is used as an argument
    for most functions, and creating one should be the first step in
    simulating the system for some parameters. """

    def __init__(self, rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions):

        assert isinstance(rows_per_batch, int)
        assert isinstance(num_servers, int)
        assert isinstance(q, int)
        assert isinstance(num_outputs, int)
        assert isinstance(server_storage, float) or server_storage == 1
        assert isinstance(num_partitions, int)
        assert server_storage <= 1, 'Server storage can at most be 1'

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

        assert num_source_rows / num_partitions % 1 == 0, \
            'There must be an integer number of source rows per partition.'

        assert num_coded_rows / num_partitions % 1 == 0, \
            'There must be an integer number of coded rows per partition. num_partitions: %d' \
            % num_partitions

        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition

        self.sq = self.s_q()

        # Setup a dict to cache delay computations in
        self.cached_delays = dict()
        self.cached_reduce_delays = dict()

        return

    @classmethod
    def fixed_complexity_parameters(cls, rows_per_server, rows_per_partition, min_num_servers, code_rate, muq):
        """ Attempt to find a set of servers with a fixed number of rows per
        server and rows per partition. """

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

        return cls(rows_per_batch, num_servers, q, q, server_storage, num_partitions)

    def __str__(self):
        string = '------------------------------\n'
        string += 'System Parameters:\n'
        string += '------------------------------\n'
        string += 'Rows per batch:  ' + str(self.rows_per_batch) + '\n'
        string += 'Servers: ' + str(self.num_servers) + '\n'
        string += 'Wait for: ' + str(self.q) + '\n'
        string += 'Number of outputs: ' + str(self.num_outputs) + '\n'
        string += 'Server storage: ' + str(self.server_storage) + '\n'
        string += 'Batches: ' + str(self.num_batches) + '\n'
        string += '|T|: ' + str(self.server_storage * self.q) + '\n'
        string += 'Code rate: ' + str(self.q / self.num_servers) + '\n'
        string += 'Source rows: ' + str(self.num_source_rows) + '\n'
        string += 'Coded rows: ' + str(self.num_coded_rows) + '\n'
        string += 'Partitions: ' + str(self.num_partitions) + '\n'
        string += 'Rows per partition: ' + str(self.rows_per_partition) + '\n'
        string += 'Multicast messages: ' + str(self.num_multicasts()) + '\n'
        string += 'Unpartitioned load: ' + str(self.unpartitioned_load()) + '\n'
        string += '------------------------------'
        return string

    def old_identifier(self):
        """ Return the old string identifier for these parameters. """
        string = 'm_' + str(self.num_source_rows)
        string += '_K_' + str(self.num_servers)
        string += '_q_' + str(self.q)
        string += '_N_' + str(self.num_outputs)
        string += '_mu_' + str(self.server_storage)
        string += '_T_' + str(self.num_partitions)
        return string

    def identifier(self):
        """ Return a string identifier for these parameters. """
        string = 'm_' + str(self.num_source_rows)
        string += '_K_' + str(self.num_servers)
        string += '_q_' + str(self.q)
        string += '_N_' + str(self.num_outputs)
        string += '_muq_' + str(int(self.server_storage * self.q))
        string += '_T_' + str(self.num_partitions)
        return string

    def s_q(self):
        """ Calculate S_q as defined in Li2016. """
        muq = int(self.server_storage * self.q)
        s = 0
        for j in range(muq, 0, -1):
            s = s + self.B_j(j)

            if s > (1 - self.server_storage):
                return max(min(j + 1, muq), 1)
            elif s == (muq):
                return max(j, 1)

        return 1

    def B_j(self, j):
        """ Calculate B_j as defined in Li2016. """
        B_j = nchoosek(self.q-1, j) * nchoosek(self.num_servers-self.q,
                                               int(self.server_storage*self.q)-j)

        B_j = B_j / (self.q / self.num_servers) / nchoosek(self.num_servers,
                                                           int(self.server_storage * self.q))
        return B_j

    def multicast_load(self):
        """ The communication load after only multicasting. """
        muq = int(self.server_storage*self.q)
        load = 0
        for j in range(self.sq, muq + 1):
            load = load + self.B_j(j) / j

        return load

    def num_multicasts(self):
        """ Total number of multicast messages. """
        return self.multicast_load() * self.num_source_rows * self.num_outputs

    def unpartitioned_load(self, enable_load_2=True):
        """ Calculate the total communication load per output vector for an
        unpartitioned storage design as defined in Li2016.

        Args:
        enable_load_2: Allow multicasting for subsets of size s_q-1

        Returns:
        Communication load per output vector
        """
        muq = int(self.server_storage * self.q)

        # Load assuming remaining values are unicast
        load_1 = 1 - self.server_storage
        for j in range(self.sq, muq + 1):
            Bj = self.B_j(j)
            load_1 = load_1 + Bj / j
            load_1 = load_1 - Bj

        # Load assuming more multicasting is done
        if self.sq > 1:
            load_2 = 0
            for j in range(self.sq - 1, muq + 1):
                load_2 = load_2 + self.B_j(j) / j
        else:
            load_2 = math.inf

        # Return the minimum if enable_load_2 is enabled. Otherwise return load_1
        if enable_load_2:
            return min(load_1, load_2)
        else:
            return load_1

    def computational_delay(self, q=None):
        """Return the delay incurred in the map phase.

        Calculates the computational delay assuming a shifted
        exponential distribution.

        Args:
        q: The number of servers to wait for. If q is None, the value in self is used.

        Returns:
        The computational delay.

        """

        assert q is None or isinstance(q, int)
        if q is None:
            q = self.q

        # Return infinity if waiting for more than num_servers servers
        if q > self.num_servers:
            return math.inf

        # Return a cached result if there is one
        if q in self.cached_delays:
            return self.cached_delays[q]

        delay = 1
        for j in range(self.num_servers - q + 1, self.num_servers):
            delay += 1 / j

        delay *= self.server_storage * self.num_outputs

        # Cache the result
        self.cached_delays[q] = delay

        return delay

    def computational_delay_ldpc(self, q=None, overhead=1.10):
        '''Estimate the theoretical computational delay when using an LDPC
        code.

        Args:
        q: The number of servers to wait for. If q is None, the value in self is used.

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

        # Return a cached result if there is one
        if num_partitions in self.cached_reduce_delays:
            return self.cached_reduce_delays[num_partitions]

        delay = 1
        for j in range(1, self.q):
            delay += 1 / j

        # TODO: Is this the correct way to calculate the packet size?
        # A given server does not necessarily experience the erasure
        # of all symbols on a machine that failed to finish.
        delay *= complexity.block_diagonal_decoding_complexity(self.num_coded_rows,
                                                               1,
                                                               1 - self.q / self.num_servers,
                                                               num_partitions)
        delay *= self.num_outputs / self.q

        # TODO: We assume that the source matrix is square for now
        delay /= complexity.matrix_vector_complexity(self.num_source_rows, self.num_source_rows)

        # Cache the result
        self.cached_reduce_delays[num_partitions] = delay

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

        # TODO: We assume that the source matrix is square for now
        delay /= complexity.matrix_vector_complexity(self.num_source_rows, self.num_source_rows)

        return delay

class Perspective(object):
    """ Representation of the count vectors from the perspective of a single server

    Depending on which servers first finish the map computation, a different
    set of batches will be involved in the multicast phase.

    Attributes:
    score: The number of additional rows required to decode all partitions.
    count: An array where count[i] is the number of symbols from partition i.
    rows:
    """

    def __init__(self, score, count, rows, perspective_id=None):
        """ Create a new Perspective object

        Args:
        score: The number of additional rows required to decode all partitions.
        count: An array where count[i] is the number of symbols from partition i.
        rows:
        perspective_id: A unique ID for this instance.
        If None, one is generated.
        """
        self.score = score
        self.count = count
        self.rows = rows

        # Generate a new ID if none was provided
        if isinstance(perspective_id, uuid.UUID):
            self.perspective_id = perspective_id
        else:
            self.perspective_id = uuid.uuid4()

    def __str__(self):
        string = 'Score: ' + str(self.score)
        string += ' Count: ' + str(self.count)
        string += ' Rows: ' + str(self.rows)
        return string

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.rows == other.rows
        return False

    def __hash__(self):
        return hash(self.perspective_id)

    def increment(self, col):
        """ Increment the column and update the indices.

        Args:
        col: The column of the assignment matrix to increment.

        Returns:
        A new Perspective instance with the updated values.
        """

        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] = count[col] + 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

    def decrement(self, col):
        """ Decrement the column and update the indices.

        Args:
        col: The column of the assignment matrix to decrement.

        Returns:
        A new Perspective instance with the updated values.
        """
        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] -= 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

class BatchResult(object):
    """ The results index from the perspective of a row of the assignment matrix """

    def __init__(self, perspectives=None, summary=None):
        if perspectives is None or summary is None:
            self.perspectives = dict()
        else:
            assert isinstance(perspectives, dict)
            assert isinstance(summary, list)
            assert isinstance(summary, list)
            self.perspectives = perspectives
            self.summary = summary

        return

    def __getitem__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        return self.perspectives[key]

    def __setitem__(self, key, value):
        if not isinstance(key, Perspective) or not isinstance(value, Perspective):
            raise TypeError('Key and value must be of type perspective')

        self.perspectives[key] = value

    def __delitem__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        del self.perspectives[key]

    def __contains__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        return key in self.perspectives


    def init_summary(self, par):
        """ Build a summary of the number of perspectives that need symbols
        from a certain partition.
        """

        num_perspectives = len(self.perspectives)

        # Build a simple summary that keeps track of how many symbols
        # that need a certain partition.
        self.summary = [num_perspectives for x in range(par.num_partitions)]

    def keys(self):
        """ Return the keys of the perspectives dict. """
        return self.perspectives.keys()

    def copy(self):
        """ Returns a shallow copy of itself """
        perspectives = dict(self.perspectives)
        summary = list(self.summary)
        return BatchResult(perspectives, summary)

class Assignment(object):
    """ Storage design representation

    Representing a storage assignment. Has support for dynamic programming
    methods to efficiently find good assignments for small instances.

    Attributes:
    assignment_matrix: An num_batches by num_partitions Numpy array, where
    assignment_matrix[i, j] is the number of rows from partition j stored in
    batch i.
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    score: The sum of all unicasts that need to be sent until all partitions
    can be decoded over all subsets that can complete the map phase. If either
    score or index is set to None, the index will be built from scratch. Set
    both to False to prevent the index from being built.
    index: A list of BatchResult of length equal to the number of batches. If
    either score or index is set to None, the index will be built from scratch.
    Set both to False to prevent the index from being built.
    """

    def __init__(self, par, assignment_matrix=None, labels=None, score=None, index=None):
        self.par = par

        if assignment_matrix is None:
            self.assignment_matrix = np.zeros([par.num_batches, par.num_partitions])
        else:
            self.assignment_matrix = assignment_matrix

        if labels is None:
            self.labels = [set() for x in range(par.num_servers)]
            self.label()
        else:
            self.labels = labels

        if score is None or index is None:
            self.build_index()
        else:
            self.score = score
            self.index = index

    def __str__(self):
        string = ''
        string += 'assignment matrix:\n'
        string += str(self.assignment_matrix) + '\n'
        string += 'labels:\n'
        string += str(self.labels) + '\n'
        string += 'Score: ' + str(self.score)
        return string

    def save(self, directory='./assignments/'):
        """ Save the assignment to disk

        Args:
        directory: Directory to save to
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + self.par.identifier() + '.npy', self.assignment_matrix)
        return

    @classmethod
    def load(cls, par, directory='./assignments/'):
        """ Load assignment from disk

        Args:
        par: System parameters
        directory: Directory to load from

        Returns:
        The loaded assignment
        """
        if directory is None:
            raise FileNotFoundError()

        assignment_matrix = np.load(directory + par.identifier() + '.npy')
        return cls(par, assignment_matrix=assignment_matrix, score=False, index=False)

    def build_index(self):
        """ Build the dynamic programming index

        Build an index pairing rows of the assignment matrix to which
        perspectives they appear in. Only run once when creating a new
        assignment.
        """

        self.score = 0

        # Index for which sets every row is contained in
        self.index = [BatchResult(self.par) for x in range(self.par.num_batches)]
        self.num_subsets = nchoosek(self.par.num_servers, self.par.q)
        subsets = it.combinations(range(self.par.num_servers), self.par.q)

        # Build an index for which count vectors every row is part of
        for Q in subsets:
            for k in Q:
                rows = set()
                for batch in self.labels[k]:
                    rows.add(batch)

                for j in range(self.par.sq, int(self.par.server_storage*self.par.q) + 1):
                    for subset in it.combinations([x for x in Q if x != k], j):
                        rows = rows | set.intersection(*[self.labels[x] for x in subset])

                selector_vector = np.zeros(self.par.num_batches)
                for row in rows:
                    selector_vector[row] = 1

                count_vector = np.dot(selector_vector, self.assignment_matrix)
                count_vector -= self.par.num_source_rows / self.par.num_partitions
                score = remaining_unicasts(count_vector)
                self.score = self.score + score

                perspective = Perspective(score, count_vector, rows)
                for row in rows:
                    assert perspective not in self.index[row]
                    self.index[row][perspective] = perspective

        # Initialize the summaries
        for batch_result in self.index:
            batch_result.init_summary(self.par)

    def label(self, shuffle=True):
        """Label the batches with server subsets

        Label all batches with subsets.

        Args:
        shuffle: Shuffle the labeling if True. Otherwise label in the
        order returned by itertools.combinations.

        """
        assert self.par.server_storage * self.par.q % 1 == 0, 'Must be integer'
        labels = list(it.combinations(range(self.par.num_servers),
                                      int(self.par.server_storage * self.par.q)))
        if shuffle:
            random.shuffle(labels)

        row = 0
        for label in labels:
            for server in label:
                self.labels[server].add(row)
            row += 1
        return

    def bound(self):
        """ Compute a bound for this assignment """

        assert self.index and self.score, 'Cannot compute bound if there is no index.'
        decreased_unicasts = 0
        for row_index in range(self.assignment_matrix.shape[0]):
            row = self.assignment_matrix[row_index]
            remaining_assignments = self.par.rows_per_batch - sum(row)
            assert remaining_assignments >= 0
            decreased_unicasts += max(self.index[row_index].summary) * remaining_assignments

        # Bound can't be less than 0
        return max(self.score - decreased_unicasts, 0)

    def increment(self, row, col):
        """ Increment assignment_matrix[row, col]

        Increment the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:
        row: The row index
        col: The column index

        Returns:
        A new assignment object.
        """

        assert self.index and self.score, 'Cannot increment if there is no index.'
        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] = assignment_matrix[row, col] + 1
        objective_value = self.score

        # Select the perspectives linked to the row
        perspectives = index[row]

        # Iterate over all perspectives linked to that row
        for perspective_key in perspectives.keys():
            perspective = perspectives[perspective_key]

            # Create a new perspective from the updated values
            new_perspective = perspective.increment(col)

            # Update the objective function
            objective_value = objective_value - (perspective.score - new_perspective.score)

            # Update the index for all rows which include this perspective
            for perspective_row in perspective.rows:
                assert hash(new_perspective) == hash(index[perspective_row][perspective])
                index[perspective_row][perspective] = new_perspective

            # Update the summaries if the count reached zero for the
            # indicated column
            if new_perspective.count[col] == 0:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] = index[perspective_row].summary[col] - 1

        # Return a new assignment object
        return Assignment(self.par,
                          assignment_matrix=assignment_matrix,
                          labels=self.labels,
                          score=objective_value,
                          index=index)

    def decrement(self, row, col):
        """ Decrement assignment_matrix [row, col]

        Decrement the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:
        row: The row index
        col: The column index

        Returns:
        A new assignment object.
        """

        assert self.index and self.score, 'Cannot decrement if there is no index.'
        assert self.assignment_matrix[row, col] >= 1, 'Can\'t decrement a value less than 1.'

        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] = assignment_matrix[row, col] - 1
        objective_value = self.score

        # Select the perspectives linked to the row
        perspectives = index[row]

        # Iterate over all perspectives linked to that row
        for perspective_key in perspectives.keys():
            perspective = perspectives[perspective_key]

            # Create a new perspective from the updated values
            new_perspective = perspective.decrement(col)

            # Update the objective function
            objective_value = objective_value - (perspective.score - new_perspective.score)

            # Update the index for all rows which include this perspective
            for perspective_row in perspective.rows:
                assert hash(new_perspective) == hash(index[perspective_row][perspective])
                index[perspective_row][perspective] = new_perspective

            # Update the summaries if the count reached zero for the
            # indicated column
            if new_perspective.count[col] == -1:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] += 1

        # Return a new assignment object
        return Assignment(self.par,
                          assignment_matrix=assignment_matrix,
                          labels=self.labels,
                          score=objective_value,
                          index=index)

    def evaluate(self, row, col):
        """ Return the performance change that incrementing (row, col)
        would induce without changing the assignment.
        """

        assert self.index and self.score, 'Cannot evaluateif there is no index.'
        return self.index[row].summary[col]

    def copy(self):
        """ Return a deep copy of the assignment. """

        assignment_matrix = np.array(self.assignment_matrix)
        if self.index:
            index = [x.copy() for x in self.index]
            score = self.score
        else:
            index = False
            score = False

        return Assignment(self.par,
                          assignment_matrix=assignment_matrix,
                          labels=self.labels,
                          score=score,
                          index=index)

def is_valid(par, assignment_matrix, verbose=False):
    """ Evaluate if an assignment is valid and complete.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    verbose: Print why an assignment might be invalid

    Returns:
    True if the assignment matrix is valid and complete. False otherwise.
    """

    for row in assignment_matrix:
        if row.sum() != par.rows_per_batch:
            if verbose:
                print('Row', row, 'does not sum to the number of rows per batch. Is:',
                      row.sum(), 'Should be:', par.rows_per_batch)
            return False

    for col in assignment_matrix.T:
        if col.sum() != par.num_coded_rows / par.num_partitions:
            if verbose:
                print('Column', col, 'does not sum to the number of rows per partition. Is',
                      col.sum(), 'Should be:', par.num_coded_rows / par.num_partitions)
            return False

    return True

def remaining_unicasts(rows_by_partition):
    """" Return the number of unicasts required to decode all partitions.

    Args:
    rows_by_partition: A numpy array of row counts by partition.

    Returns:
    The number of unicasts required to decode all partitions.
    """

    unicasts = 0
    for num_rows in rows_by_partition:
        if num_rows < 0:
            unicasts = unicasts - num_rows

    return unicasts
