'''This assignment is backed by a dense matrix and is suitable for
smaller systems.

'''
import os
import logging
import random
import itertools
import numpy as np
import model
from assignments import Assignment

class DenseAssignment(Assignment):
    '''Dense storage design representation

    Representing a storage assignment. This implementation is backed
    by a dense matrix.

    Attributes:

    assignment_matrix: An num_batches by num_partitions Numpy array,
    where assignment_matrix[i, j] is the number of rows from partition
    j stored in batch i.

    labels: List of sets, where the elements of the set labels[i] are
    the rows of the assignment_matrix stored at server i.

    '''

    def __init__(self, par, gamma=0, labels=None):
        '''Initialize a sparse assignment.

        Args:
        par: A parameters object.

        gamma: Number of coded rows for each partition stored in all
        batches.

        labels: List of sets, where the elements of the set labels[i]
        are the rows of the assignment_matrix stored at server i. A
        new one is generated in this is None.

        '''
        assert isinstance(par, model.SystemParameters)
        assert isinstance(gamma, int)
        assert isinstance(labels, list) or labels is None

        self.par = par
        self.gamma = gamma
        self.assignment_matrix = np.zeros((par.num_batches, par.num_partitions)) + gamma
        if labels is None:
            self.labels = [set() for x in range(par.num_servers)]
            self.label()
        else:
            self.labels = labels

        return

    def __repr__(self):
        string = ''
        string += 'assignment matrix:\n'
        string += str(self.assignment_matrix) + '\n'
        string += 'gamma:\n'
        string += str(self.gamma) + '\n'
        string += 'labels:\n'
        string += str(self.labels) + '\n'
        return string

    def batch_union(self, batch_indices):
        '''Compute the union of symbols stored in a set of batches.

        Args:
        batch_indices: Iterable of batch indices.

        Returns: A dense Numpy array containing the counts of symbols
        stored in a union of batches.

        '''
        symbols = np.zeros(self.par.num_partitions)
        for batch_index in batch_indices:
            symbols += self.assignment_matrix[batch_index]

        return symbols

    def save(self, directory='./saved_assignments/'):
        """ Save the assignment to disk

        Args:
        directory: Directory to save to
        """
        raise NotImplemented

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + self.par.identifier() + '.npy', self.assignment_matrix)
        return

    @classmethod
    def load(cls, par, directory='./saved_assignments/'):
        """ Load assignment from disk

        Args:
        par: System parameters
        directory: Directory to load from

        Returns:
        The loaded assignment
        """
        raise NotImplemented
        if directory is None:
            raise FileNotFoundError()

        assignment_matrix = np.load(directory + par.identifier() + '.npy')
        return cls(par, assignment_matrix=assignment_matrix, score=False, index=False)

    def label(self, shuffle=False):
        '''Label the batches with server subsets

        Label all batches with subsets.

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

    def increment(self, rows, cols, data):
        '''Increment assignment_matrix[rows[i], cols[i]] by data[i] for all i

        Args:
        row: List of row indices

        col: List of column indices

        data: List of values to increment by

        Returns: Returns self. Does not copy the assignment.

        '''
        assert isinstance(rows, list)
        assert isinstance(cols, list)
        assert isinstance(data, list)
        assert len(rows) == len(cols)
        assert len(cols) == len(data)
        for row, col, value in zip(rows, cols, data):
            self.assignment_matrix[row, col] += value
        return self

    def decrement(self, rows, cols, data):
        '''Decrement assignment_matrix[rows[i], cols[i]] by data[i] for all i

        Args:
        row: List of row indices

        col: List of column indices

        data: List of values to increment by

        Returns: Returns self. Does not copy the assignment.

        '''
        assert isinstance(rows, list)
        assert isinstance(cols, list)
        assert isinstance(data, list)
        assert len(rows) == len(cols)
        assert len(cols) == len(data)
        for row, col, value in zip(rows, cols, data):
            self.assignment_matrix[row, col] -= value
        return self

    def is_valid(self):
        '''Test if the assignment is valid.

        Returns: True if the assignment matrix is valid and complete,
        and False otherwise.

        '''

        correct_row_sum = self.par.rows_per_batch
        for row in self.assignment_matrix:
            if row.sum() != correct_row_sum:
                logging.warning('Sum of row\n%s \nis %d, but should be %d.',
                                str(row), row.sum(), correct_row_sum)
                return False

        correct_col_sum = self.par.num_coded_rows / self.par.num_partitions
        for col in self.assignment_matrix.T:
            if col.sum() != correct_col_sum:
                logging.warning('Sum of col\n%s \nis %d, but should be %d.',
                                str(col), col.sum(), correct_col_sum)
                return False

        return True
