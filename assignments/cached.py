'''This assignment is backed by a dense matrix and uses a dynamic
programming approach for speeding up branch-and-bound opertaions.
Suitable only for very small systems as the dynamic programming
approach needs a lot of memory.

'''
import os
import uuid
import logging
import random
import itertools
import numpy as np
from scipy.misc import comb as nchoosek
import model
from assignments import Assignment

class Perspective(object):
    '''Representation of the count vectors from the perspective of a
    single server

    Depending on which servers first finish the map computation, a
    different set of batches will be involved in the multicast phase.

    Attributes:

    score: The number of additional rows required to decode all
    partitions.

    count: An array where count[i] is the number of symbols from
    partition i.

    '''
    def __init__(self, score, count, rows, perspective_id=None):
        '''Create a new Perspective object

        Args:

        score: The number of additional rows required to decode all partitions.

        count: An array where count[i] is the number of symbols from partition i.

        rows: TODO: Write description.

        perspective_id: A unique ID for this instance. If None, one is
        generated.

        '''
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
        '''Increment the column and update the indices.

        Args:

        col: The column of the assignment matrix to increment.

        Returns: A new Perspective instance with the updated values.

        '''

        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] = count[col] + 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

    def decrement(self, col):
        '''Decrement the column and update the indices.

        Args:

        col: The column of the assignment matrix to decrement.

        Returns: A new Perspective instance with the updated values.

        '''
        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] -= 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

class BatchResult(object):
    '''The results index from the perspective of a row of the assignment
    matrix

    '''

    def __init__(self, perspectives=None, summary=None):
        if perspectives is None or summary is None:
            self.perspectives = dict()
        else:
            assert isinstance(perspectives, dict)
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
        '''Build a summary of the number of perspectives that need symbols
        from a certain partition.

        '''

        num_perspectives = len(self.perspectives)

        # Build a simple summary that keeps track of how many symbols
        # that need a certain partition.
        self.summary = [num_perspectives for x in range(par.num_partitions)]

    def keys(self):
        '''Return the keys of the perspectives dict.'''
        return self.perspectives.keys()

    def copy(self):
        '''Returns a shallow copy of itself'''
        perspectives = dict(self.perspectives)
        summary = list(self.summary)
        return BatchResult(perspectives, summary)

class CachedAssignment(Assignment):
    '''Cached storage design representation

    Representing a storage assignment. Has support for dynamic
    programming methods to efficiently find good assignments for small
    instances.

    Attributes:

    assignment_matrix: An num_batches by num_partitions Numpy array,
    where

    assignment_matrix[i, j] is the number of rows from partition j
    stored in batch i.

    gamma: Number of coded rows for each partition stored in all
    batches.

    labels: List of sets, where the elements of the set labels[i] are
    the rows of the assignment_matrix stored at server i.

    score: The sum of all unicasts that need to be sent until all
    partitions can be decoded over all subsets that can complete the
    map phase. If either score or index is set to None, the index will
    be built from scratch. Set both to False to prevent the index from
    being built.

    index: A list of BatchResult of length equal to the number of
    batches. If either score or index is set to None, the index will
    be built from scratch. Set both to False to prevent the index from
    being built.

    '''
    def __init__(self, par, gamma=0, assignment_matrix=None, labels=None, score=None, index=None):
        assert isinstance(par, model.SystemParameters)
        assert isinstance(gamma, int) and gamma >= 0
        self.par = par
        self.gamma = gamma

        if assignment_matrix is None:
            self.assignment_matrix = np.zeros([par.num_batches, par.num_partitions], dtype=np.int16)
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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return np.array_equal(self.assignment_matrix, other.assignment_matrix)

    def __str__(self):
        string = 'CachedAssignment.'
        string += ' Gamma: ' + str(self.gamma)
        string += ' Score: ' + str(self.score) + '\n'
        string += ' Matrix:\n'
        string += str(self.assignment_matrix) + '\n'
        string += 'labels: '
        string += str(self.labels) + '\n'
        return string

    def batch_union(self, batch_indices):
        '''Compute the union of symbols stored in a set of batches.

        Args:
        batch_indices: Iterable of batch indices.

        Returns: A dense Numpy array containing the counts of symbols
        stored in a union of batches.

        '''
        row_indices = list(batch_indices)
        sorted(row_indices)
        symbols = self.assignment_matrix[row_indices, :].sum(axis=0)
        symbols += self.gamma * len(row_indices)
        return symbols

    def rows_iterator(self):
        '''Iterate over the rows of the assignment matrix.'''
        for row in self.assignment_matrix:
            yield row

        return

    def save(self, directory='./saved_assignments/'):
        '''Save the assignment to disk

        Args:

        directory: Directory to save to

        '''
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + self.par.identifier() + '.npy', self.assignment_matrix)
        return

    @classmethod
    def load(cls, par, directory='./saved_assignments/'):
        '''Load assignment from disk

        Args:

        par: System parameters

        directory: Directory to load from

        Returns: The loaded assignment

        '''
        if directory is None:
            raise FileNotFoundError()

        assignment_matrix = np.load(directory + par.identifier() + '.npy')
        return cls(par, assignment_matrix=assignment_matrix, score=False, index=False)

    def build_index(self):
        '''Build the dynamic programming index

        Build an index pairing rows of the assignment matrix to which
        perspectives they appear in. Only run once when creating a new
        assignment.

        '''
        self.score = 0

        # Index for which sets every row is contained in
        self.index = [BatchResult(self.par) for _ in range(self.par.num_batches)]
        self.num_subsets = nchoosek(self.par.num_servers, self.par.q, exact=True)
        subsets = itertools.combinations(range(self.par.num_servers), self.par.q)

        # Build an index for which count vectors every row is part of
        for Q in subsets:
            for k in Q:
                rows = set()
                for batch in self.labels[k]:
                    rows.add(batch)

                for j in range(self.par.multicast_set_size_1(), int(self.par.server_storage*self.par.q) + 1):
                    for subset in itertools.combinations([x for x in Q if x != k], j):
                        rows = rows | set.intersection(*[self.labels[x] for x in subset])

                selector_vector = np.zeros(self.par.num_batches)
                for row in rows:
                    selector_vector[row] = 1

                count_vector = np.dot(selector_vector, self.assignment_matrix)
                count_vector -= self.par.num_source_rows / self.par.num_partitions
                #print(count_vector, count_vector + self.gamma*len(rows))
                count_vector += self.gamma * len(rows)
                #print(count_vector)
                score = remaining_unicasts(count_vector)
                self.score = self.score + score

                perspective = Perspective(score, count_vector, rows)
                for row in rows:
                    assert perspective not in self.index[row]
                    self.index[row][perspective] = perspective

        # Initialize the summaries
        for batch_result in self.index:
            batch_result.init_summary(self.par)

    def label(self, shuffle=False):
        '''Label the batches with server subsets

        Args:

        shuffle: Shuffle the labeling if True. Otherwise label in the
        order returned by itertools.combinations.

        '''
        assert self.par.server_storage * self.par.q % 1 == 0, 'Must be integer'
        labels = list(itertools.combinations(range(self.par.num_servers),
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
        '''Compute a bound for this assignment'''
        assert self.index and self.score, 'Cannot compute bound if there is no index.'
        decreased_unicasts = 0
        for row_index in range(self.assignment_matrix.shape[0]):
            row = self.assignment_matrix[row_index] + self.gamma
            remaining_assignments = self.par.rows_per_batch - sum(row)
            assert remaining_assignments >= 0, remaining_assignments
            decreased_unicasts += max(self.index[row_index].summary) * remaining_assignments

        # Bound can't be less than 0
        return max(self.score - decreased_unicasts, 0)

    def increment(self, rows, cols, data):
        '''Increment assignment_matrix[rows[i], cols[i]] by data[i] for all i.
        TODO: This can be made more efficient by not copying the index
        for each element etc.

        Args:
        row: List of row indices

        col: List of column indices

        data: List of values to increment by

        Returns: A new assignment object.

        '''
        assert isinstance(rows, list)
        assert isinstance(cols, list)
        assert isinstance(data, list)
        assert len(rows) == len(cols)
        assert len(cols) == len(data)
        assignment = self
        for row, col, value in zip(rows, cols, data):
            for _ in range(value):
                assignment = assignment.increment_element(row, col)

        return assignment

    def decrement(self, rows, cols, data):
        '''Decrement assignment_matrix[rows[i], cols[i]] by data[i] for all i.
        TODO: This can be made more efficient by not copying the index
        for each element etc.

        Args:
        row: List of row indices

        col: List of column indices

        data: List of values to increment by

        Returns: A new assignment object.

        '''
        assert isinstance(rows, list)
        assert isinstance(cols, list)
        assert isinstance(data, list)
        assert len(rows) == len(cols)
        assert len(cols) == len(data)
        assignment = self
        for row, col, value in zip(rows, cols, data):
            for _ in range(value):
                assignment = assignment.decrement_element(row, col)

        return assignment

    def increment_element(self, row, col):
        '''Increment assignment_matrix[row, col]

        Increment the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:

        row: The row index

        col: The column index

        Returns: A new assignment object.

        '''
        assert self.index and self.score, 'Cannot increment if there is no index.'

        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] += 1
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
        return CachedAssignment(self.par, gamma=self.gamma,
                                assignment_matrix=assignment_matrix,
                                labels=self.labels,
                                score=objective_value,
                                index=index)

    def decrement_element(self, row, col):
        '''Decrement assignment_matrix [row, col]

        Decrement the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:

        row: The row index

        col: The column index

        Returns: A new assignment object.

        '''

        assert self.index and self.score, 'Cannot decrement if there is no index.'
        assert self.assignment_matrix[row, col] >= 1, 'Can\'t decrement a value less than 1.'

        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] -= 1
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
        return CachedAssignment(self.par, gamma=self.gamma,
                                assignment_matrix=assignment_matrix,
                                labels=self.labels,
                                score=objective_value,
                                index=index)

    def evaluate(self, row, col):
        '''Return the performance change that incrementing (row, col) would
        induce without changing the assignment.

        '''

        assert self.index and self.score, 'Cannot evaluate if there is no index.'
        return self.index[row].summary[col]

    def copy(self):
        '''Return a deep copy of the assignment.'''

        assignment_matrix = np.array(self.assignment_matrix)
        if self.index:
            index = [x.copy() for x in self.index]
            score = self.score
        else:
            index = False
            score = False

        return CachedAssignment(self.par, gamma=self.gamma,
                                assignment_matrix=assignment_matrix,
                                labels=self.labels,
                                score=score,
                                index=index)

    def is_valid(self):
        '''Test if the assignment is valid.

        Returns: True if the assignment matrix is valid and complete,
        and False otherwise.

        '''

        correct_row_sum = round(self.par.rows_per_batch - self.par.num_partitions * self.gamma)
        for i in range(self.par.num_batches):
            row = self.assignment_matrix[i, :]
            if row.sum() != correct_row_sum:
                logging.warning('Sum of row %d\n%s \nis %d, but should be %d.',
                                i, str(row), row.sum(), correct_row_sum)
                return False

        correct_col_sum = round(self.par.num_coded_rows / self.par.num_partitions - self.par.num_batches * self.gamma)
        for i in range(self.par.num_partitions):
            col = self.assignment_matrix.T[i, :]
            if col.sum() != correct_col_sum:
                logging.warning('Sum of column %d\n%s \nis %d, but should be %d.',
                                i, str(col), col.sum(), correct_col_sum)
                return False

        return True

def remaining_unicasts(rows_by_partition):
    '''' Return the number of unicasts required to decode all partitions.

    Args:

    rows_by_partition: A numpy array of row counts by partition.

    Returns: The number of unicasts required to decode all partitions.

    '''
    # TODO: Perhaps have this function as an argument instead?

    unicasts = 0
    for num_rows in rows_by_partition:
        if num_rows < 0:
            unicasts -= num_rows

    return unicasts
