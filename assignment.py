'''This module contains the representations of assignments of coded
rows into batches.

'''

import logging
import random
import itertools
from abc import ABC, abstractmethod
import scipy as sp
import numpy as np
import model

class Assignment(ABC):
    '''Abstract superclass representing an assignment of encoded rows into
    batches.

    '''
    @abstractmethod
    def increment(self, row, col):
        pass

    @abstractmethod
    def decrement(self, row, col):
        pass

    @abstractmethod
    def batch_union(batch_indices):
        pass

    @abstractmethod
    def save(self, directory='./assignments/'):
        pass

    @abstractmethod
    def load(self, par, directory='./assignments/'):
        pass

class SparseAssignment(Assignment):
    '''Sparse storage design representation

    Representing a storage assignment. This implementation is backed
    by two sparse matrices.

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
        self.assignment_matrix = sp.sparse.csr_matrix((par.num_batches, par.num_partitions))
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
        string += str(gamma) + '\n'
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

        symbols += self.gamma * len(batch_indices)
        return np.array(symbols)[0]

    def save(self, directory='./assignments/'):
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
    def load(cls, par, directory='./assignments/'):
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

    def label(self, shuffle=True):
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

    def increment(self, row, col):
        '''Increment assignment_matrix[row, col]

        Args:
        row: The row index
        col: The column index

        Returns: Returns self. Does not copy the assignment.

        '''
        assert isinstance(row, int) and 0 <= row < self.par.num_batches
        assert isinstance(col, int) and 0 <= col < self.par.num_partitions
        self.assignment_matrix[row, col] += 1
        return self

    def decrement(self, row, col):
        '''Decrement assignment_matrix[row, col]

        Args:
        row: The row index
        col: The column index

        Returns: Returns self. Does not copy the assignment.

        '''
        assert isinstance(row, int) and 0 <= row < self.par.num_batches
        assert isinstance(col, int) and 0 <= col < self.par.num_partitions
        self.assignment_matrix[row, col] -= 1
        return self

    def is_valid(self):
        '''Test if the assignment is valid.

        Returns: True if the assignment matrix is valid and complete,
        and False otherwise.

        '''

        correct_row_sum = self.par.rows_per_batch - self.par.num_partitions * self.gamma
        for row in self.assignment_matrix:
            if row.sum() != correct_row_sum:
                logging.debug('Sum of row %s is %d, but should be %d.',
                              str(row), row.sum(), correct_row_sum)
                return False

        correct_col_sum = self.par.num_coded_rows / self.par.num_partitions - self.par.num_batches * self.gamma
        for col in self.assignment_matrix.T:
            if col.sum() != correct_col_sum:
                logging.debug('Sum of col %s is %d, but should be %d.',
                              str(col), col.sum(), correct_col_sum)
                return False

        return True
