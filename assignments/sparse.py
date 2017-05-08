'''This assignment is backed by a sparse matrix and is suitable for
large systems.

'''

import os
import logging
import random
import itertools
import scipy as sp
import numpy as np
import model
from assignments import Assignment

class SparseAssignment(Assignment):
    '''Sparse storage design representation

    Representing a storage assignment. This implementation is backed
    by a sparse matrix and a scalar representing a value common to all
    elements.

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
        self.assignment_matrix = sp.sparse.coo_matrix((par.num_batches, par.num_partitions), dtype=np.int8)
        self.assignment_matrix_csr = None
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

    def __str__(self):
        if self.assignment_matrix_csr is None:
            self.assignment_matrix_csr = self.assignment_matrix.tocsr()
        string = ''
        string += 'gamma = ' + str(self.gamma) + '\n'

        rows_to_print = min(100, self.assignment_matrix_csr.shape[0])
        matrix = self.assignment_matrix_csr.A[0:rows_to_print]
        for row_index in range(rows_to_print):
            string += str(self.batch_labels[row_index]) + ' '
            string += '['
            row = matrix[row_index]
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
        assert isinstance(batch_indices, set)
        if self.assignment_matrix_csr is None:
            self.assignment_matrix_csr = self.assignment_matrix.tocsr()
        row_indices = list(batch_indices)
        sorted(row_indices)
        symbols_slice = self.assignment_matrix_csr.A[row_indices, :]
        symbols = symbols_slice.sum(axis=0)
        symbols += self.gamma * len(batch_indices)
        return symbols

    def rows_iterator(self):
        '''Iterate over the rows of the assignment matrix.'''
        if self.assignment_matrix_csr is None:
            self.assignment_matrix_csr = self.assignment_matrix.tocsr()

        for row in self.assignment_matrix_csr.A:
            yield row

        return

    def batch_union_sparse(self, batch_indices):
        '''Compute the union of symbols stored in a set of batches.

        Args:
        batch_indices: Iterable of batch indices.

        Returns: A dense Numpy array containing the counts of symbols
        stored in a union of batches.

        '''
        raise NotImplemented

        assert isinstance(batch_indices, set)
        if self.assignment_matrix_csr is None:
            self.assignment_matrix_csr = self.assignment_matrix.tocsr()

        cols = list(batch_indices)
        rows = [0] * len(cols)
        data = [1] * len(cols)
        #selection_vector = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1, self.par.num_batches))
        # selection_vector = sp.sparse.coo_matrix((data, (rows, cols)),
        #                                         shape=(1, self.par.num_batches))

        # symbols = selection_vector.dot(self.assignment_matrix).A[0]

        selection_vector = np.zeros((1, self.par.num_batches))
        selection_vector[:, cols] = 1
        symbols = (selection_vector * self.assignment_matrix_csr)

        #print(self.par.num_partitions, selection_vector.shape, self.assignment_matrix.shape, symbols.shape)
        symbols += self.gamma * len(batch_indices)
        return symbols[0]

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
        self.batch_labels = list(itertools.combinations(range(self.par.num_servers),
                                                        int(self.par.server_storage * self.par.q)))

        if shuffle:
            random.shuffle(self.batch_labels)

        row = 0
        for label in self.batch_labels:
            for server in label:
                self.labels[server].add(row)
            row += 1
        return

    def increment(self, rows, cols, data):
        '''Increment assignment_matrix[rows[i], cols[i]] by data[i] for all i.

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
        self.assignment_matrix += sp.sparse.coo_matrix((data, (rows, cols)), dtype=np.int16,
                                                       shape=self.assignment_matrix.shape)
        self.assignment_matrix_csr = None

        # Eliminate duplicate entries
        self.assignment_matrix.sum_duplicates()
        return self

    def decrement(self, rows, cols, data):
        '''Decrement assignment_matrix[rows[i], cols[i]] by data[i] for all i.

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
        self.assignment_matrix -= sp.sparse.coo_matrix((data, (rows, cols)), dtype=np.int8,
                                                       shape=self.assignment_matrix.shape)

        # Eliminate duplicate entries
        self.assignment_matrix.sum_duplicates()
        return self

    def is_valid(self):
        '''Test if the assignment is valid.

        Returns: True if the assignment matrix is valid and complete,
        and False otherwise.

        '''

        correct_row_sum = round(self.par.rows_per_batch - self.par.num_partitions * self.gamma)
        for i in range(self.par.num_batches):
            row = self.assignment_matrix.getrow(i)
            if row.sum() != correct_row_sum:
                logging.warning('Sum of row %d\n%s \nis %d, but should be %d.',
                                i, str(row), row.sum(), correct_row_sum)
                return False

        correct_col_sum = round(self.par.num_coded_rows / self.par.num_partitions - self.par.num_batches * self.gamma)
        for i in range(self.par.num_partitions):
            col = self.assignment_matrix.getcol(i)
            if col.sum() != correct_col_sum:
                logging.warning('Sum of column %d\n%s \nis %d, but should be %d.',
                                i, str(col), col.sum(), correct_col_sum)
                return False

        return True
