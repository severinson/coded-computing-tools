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
from assignments import Assignment, AssignmentError

class SparseAssignmentError(AssignmentError):
    '''Base class for exceptions thrown by this module.'''

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

    def increment(self, cols, values):
        '''Increment the column and update the indices.

        Args:

        cols: List of columns indices to increment.

        values: Values to increment by.

        Returns: A new Perspective instance with the updated values.

        '''

        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and update the score
        count[cols] += values
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

    def decrement(self, cols, values):
        '''Decrement the column and update the indices.

        Args:

        cols: List of columns indices to decrement.

        values: Values to decrement by.

        Returns: A new Perspective instance with the updated values.

        '''
        return self.increment(cols, [-value for value in values])

        # # Copy the count array
        # count = np.array(self.count)

        # # Increment the indicated column and calculate the updated score
        # count[col] -= 1
        # score = remaining_unicasts(count)

        # # Return a new instance with the updated values
        # return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

class BatchResult(object):
    '''The results index from the perspective of a row of the assignment
    matrix

    '''

    def __init__(self, perspectives=None, summary=None):
        if perspectives is None or summary is None:
            self.perspectives = dict()
        else:
            assert isinstance(perspectives, dict)
            assert isinstance(summary, np.ndarray)
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
        self.summary = np.asarray([num_perspectives] * par.num_partitions, dtype=np.int16)

    def keys(self):
        '''Return the keys of the perspectives dict.'''
        return self.perspectives.keys()

    def copy(self):
        '''Returns a copy of itself'''
        perspectives = dict(self.perspectives)
        summary = np.array(self.summary)
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

    def prettystr(self):
        '''Pretty print the this object.'''
        string = ''
        string += 'gamma = ' + str(self.gamma) + '\n'

        rows_to_print = min(100, self.assignment_matrix.shape[0])
        for row_index in range(rows_to_print):
            string += '['
            row = self.assignment_matrix[row_index]
            for col in row:
                if col == 0:
                    string += ' '
                else:
                    string += '.'

            string += ']\n'
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

        filename = os.path.join(directory, self.par.identifier() + '.npy')
        np.save(filename, self.assignment_matrix)
        return

    @classmethod
    def load(cls, parameters, directory='./saved_assignments/'):
        '''Load assignment from disk

        Args:

        parameters: System parameters

        directory: Directory to load from

        Returns: The loaded assignment

        '''
        if directory is None:
            raise FileNotFoundError()

        filename = os.path.join(directory, parameters.identifier() + '.npy')
        assignment_matrix = np.zeros(
            [parameters.num_batches, parameters.num_partitions],
            dtype=np.int16
        )
        assignment_matrix[:] = np.load(filename)

        # Infer gamma
        gamma = parameters.rows_per_batch
        gamma -= assignment_matrix[0].sum()
        gamma /= parameters.num_partitions
        if gamma % 1 != 0:
            raise CachedAssignmentError('Could not infer the value of gamma.')

        return cls(parameters, gamma=int(gamma), assignment_matrix=assignment_matrix,
                   score=False, index=False)

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
            decreased_unicasts += self.index[row_index].summary.max() * remaining_assignments

        # Bound can't be less than 0
        return max(self.score - decreased_unicasts, 0)

    def increment(self, row_indices, col_indices, values):
        '''Increment assignment_matrix[row_indices[i], col_indices[i]] by
        values[i] for all i.

        Args:

        row_indices: List of row indices

        col_indices: List of column indices

        values: List of values to increment by

        Returns: A new assignment object.

        '''
        assert isinstance(row_indices, list)
        assert isinstance(col_indices, list)
        assert isinstance(values, list)
        assert len(row_indices) == len(col_indices)
        assert len(col_indices) == len(values)

        # Copy dynamic programming index
        index = [x.copy() for x in self.index]

        # Copy assignment matrix and objective value
        assignment_matrix = np.array(self.assignment_matrix)
        objective_value = self.score

        # Preprocess indices to eliminate duplicates
        rows = dict()
        for row, col, value in zip(row_indices, col_indices, values):
            if row in rows:
                cols = rows[row]
            else:
                cols = dict()
                rows[row] = cols

            if col in cols:
                cols[col] += value
            else:
                cols[col] = value

        # Increment
        for row, cols in rows.items():
            for col, value in cols.items():
                assignment_matrix[row, col] += value

        # Update the index
        for row in rows:

            # Select the perspectives linked to each row
            perspectives = index[row]

            # Iterate over all perspectives linked to that row
            for perspective_key in perspectives.keys():
                perspective = perspectives[perspective_key]

                # Create a new perspective from the updated values
                # cols, values = rows[row].items()
                cols = list(rows[row].keys())
                values = list(rows[row].values())
                new_perspective = perspective.increment(cols, values)

                # Update the objective function
                objective_value = objective_value - (perspective.score - new_perspective.score)

                # Update the index for all rows which include this perspective
                for perspective_row in perspective.rows:
                    index[perspective_row][perspective] = new_perspective

                # Find partitions that have saturated by checking the sign
                changed = np.zeros(self.par.num_partitions, dtype=np.int16)
                changed -= (perspective.count < 0) * (new_perspective.count >= 0)
                changed += (perspective.count >= 0) * (new_perspective.count < 0)

                # Update summaries if count reached zero for the
                # indicated column
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary += changed

        # Return a new assignment object
        return CachedAssignment(self.par, gamma=self.gamma,
                                assignment_matrix=assignment_matrix,
                                labels=self.labels,
                                score=objective_value,
                                index=index)

    def decrement(self, rows, cols, values):
        '''Decrement assignment_matrix[rows[i], cols[i]] by values[i] for all i.

        Args:

        row: List of row indices

        col: List of column indices

        values: List of values to increment by

        Returns: A new assignment object.

        '''
        assert isinstance(rows, list)
        assert isinstance(cols, list)
        assert isinstance(values, list)
        assert len(rows) == len(cols)
        assert len(cols) == len(values)
        return self.increment(rows, cols, [-value for value in values])

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

        correct_row_sum = self.par.rows_per_batch
        correct_row_sum -= self.par.num_partitions * self.gamma
        correct_row_sum = round(correct_row_sum)
        for i in range(self.par.num_batches):
            row = self.assignment_matrix[i, :]
            if row.sum() != correct_row_sum:
                logging.debug('Sum of row %d\n%s \nis %d, but should be %d.',
                              i, str(row), row.sum(), correct_row_sum)
                return False

        correct_col_sum = self.par.num_coded_rows / self.par.num_partitions
        correct_col_sum -= self.par.num_batches * self.gamma
        correct_col_sum = round(correct_col_sum)
        for i in range(self.par.num_partitions):
            col = self.assignment_matrix.T[i, :]
            if col.sum() != correct_col_sum:
                logging.debug('Sum of column %d\n%s \nis %d, but should be %d.',
                              i, str(col), col.sum(), correct_col_sum)
                return False

        return True

def remaining_unicasts(rows_by_partition):
    '''' Return the number of unicasts required to decode all partitions.

    Args:

    rows_by_partition: A numpy array of row counts by partition.

    Returns: The number of unicasts required to decode all partitions.

    '''

    unicasts = 0
    for num_rows in rows_by_partition:
        if num_rows < 0:
            unicasts -= num_rows

    return unicasts
