'''Tests of the assignments package.'''

import unittest
import itertools
import tempfile
import numpy as np
import model
from assignments.cached import CachedAssignment
from assignments.sparse import SparseAssignment

class SparseTests(unittest.TestCase):
    '''Tests fort he sparse assignment module.'''

    def test_labels(self):
        '''Verify that the labeling is valid.'''
        par = self.get_parameters()
        assignment = SparseAssignment(par)
        batches_per_server = par.server_storage * par.num_source_rows / par.rows_per_batch
        for batches in assignment.labels:
            self.assertEqual(len(batches), batches_per_server)
        return

    def test_save_load(self):
        '''Verify that saving and loading works.'''
        par = self.get_parameters_2()
        assignment = SparseAssignment(par)

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch] * par.num_batches
        assignment = assignment.increment(rows, cols, data)

        with tempfile.TemporaryDirectory() as tmpdirname:
            assignment.save(directory=tmpdirname)
            assignment_loaded = SparseAssignment.load(par, directory=tmpdirname)
            self.assertTrue(assignment.is_valid())

        return

    def test_load(self):
        '''Verify that loading non-existent files throws the correct
        exception.

        '''
        par = self.get_parameters()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(FileNotFoundError):
                assignment_loaded = SparseAssignment.load(par, directory=tmpdirname)
        return

    def test_valid(self):
        '''Test the dynamic programming index.'''
        par = self.get_parameters_2()
        assignment = CachedAssignment(par)

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch] * par.num_batches
        assignment = assignment.increment(rows, cols, data)
        self.assertTrue(assignment.is_valid())
        return

    def get_parameters(self):
        '''Get some test parameters.'''
        return model.SystemParameters(2, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)

    def get_parameters_2(self):
        '''Get some test parameters.'''
        return model.SystemParameters(6, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)

class CachedTests(unittest.TestCase):
    '''Tests for the cached assignment module.'''

    def test_is_valid(self):
        '''Verify that is_valid works correctly.'''
        par = self.get_parameters()
        assignment = CachedAssignment(par, score=False, index=False)
        assignment.assignment_matrix = np.array([[2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2],
                                                 [2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2],
                                                 [2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2]])

        self.assertTrue(assignment.is_valid())
        assignment.assignment_matrix[0, 0] = 0
        self.assertFalse(assignment.is_valid())
        return

    def test_labels(self):
        '''Verify that the labeling is valid.'''
        par = self.get_parameters()
        assignment = CachedAssignment(par, score=False, index=False)
        batches_per_server = par.server_storage * par.num_source_rows / par.rows_per_batch
        for batches in assignment.labels:
            self.assertEqual(len(batches), batches_per_server)
        return

    def test_save_load(self):
        '''Verify that saving and loading works.'''
        par = self.get_parameters()
        assignment = CachedAssignment(par)
        assignment.assignment_matrix = np.array([[2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2],
                                                 [2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2],
                                                 [2, 0, 0, 0, 0],
                                                 [0, 2, 0, 0, 0],
                                                 [0, 0, 2, 0, 0],
                                                 [0, 0, 0, 2, 0],
                                                 [0, 0, 0, 0, 2]])

        with tempfile.TemporaryDirectory() as tmpdirname:
            assignment.save(directory=tmpdirname)
            assignment_loaded = CachedAssignment.load(par, directory=tmpdirname)
            self.assertTrue(np.array_equal(assignment.assignment_matrix, assignment_loaded.assignment_matrix))

        return

    def test_load(self):
        '''Verify that loading non-existent files throws the correct
        exception.

        '''
        par = self.get_parameters()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(FileNotFoundError):
                assignment_loaded = CachedAssignment.load(par, directory=tmpdirname)
        return

    def test_copy(self):
        '''Test the assignment copying.'''
        par = self.get_parameters()
        assignment = CachedAssignment(par)
        assignment_copy = assignment.copy()
        self.assertEqual(assignment, assignment_copy)

    def test_increment_decrement(self):
        '''Test assignment increment/decrement.'''
        par = self.get_parameters()
        assignment = CachedAssignment(par)
        assignment_copy = assignment.copy()
        inc_assignment = assignment.increment([0], [0], [1])
        self.assertEqual(assignment, assignment_copy)
        self.assertNotEqual(assignment, inc_assignment)

        dec_assignment = inc_assignment.decrement([0], [0], [1])
        self.assertEqual(assignment, dec_assignment)
        return

    def test_index(self):
        '''Test the dynamic programming index.'''
        par = self.get_parameters_2()
        assignment = CachedAssignment(par)
        self.assertEqual(assignment.evaluate(0, 0), 32)

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch] * par.num_batches
        assignment = assignment.increment(rows, cols, data)
        self.assertTrue(assignment.is_valid())
        self.assertEqual(assignment.evaluate(0, 0), 9)
        return

    def test_bound(self):
        '''Test the dynamic programming bound.'''
        par = self.get_parameters_2()
        assignment = CachedAssignment(par)
        self.assertEqual(assignment.bound(), 720)

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch] * par.num_batches
        assignment = assignment.increment(rows, cols, data)
        self.assertTrue(assignment.is_valid())
        self.assertEqual(assignment.bound(), 906)
        return

    def test_score(self):
        '''Test the dynamic programming score.'''
        par = self.get_parameters_2()
        assignment = CachedAssignment(par)
        self.assertEqual(assignment.score, 3600)

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch] * par.num_batches
        assignment = assignment.increment(rows, cols, data)
        self.assertTrue(assignment.is_valid())
        self.assertEqual(assignment.score, 906)
        return

    def test_gamma(self):
        '''Test the dynamic programming for non-zero gamma.'''
        gamma = 1
        par = self.get_parameters_2()
        assignment = CachedAssignment(par)
        rows = list()
        cols = list()
        for row in range(par.num_batches):
            for col in range(par.num_partitions):
                rows.append(row)
                cols.append(col)

        data = [gamma] * len(rows)
        assignment = assignment.increment(rows, cols, data)
        gamma_assignment = CachedAssignment(par, gamma=gamma)
        self.assertEqual(assignment.evaluate(0, 0), gamma_assignment.evaluate(0, 0))
        self.assertEqual(assignment.bound(), gamma_assignment.bound())
        self.assertEqual(assignment.score, gamma_assignment.score)
        return

        # Complete the assignment
        rows = list(range(par.num_batches))
        cols = list(range(par.num_partitions)) * int(par.num_batches / par.num_partitions)
        data = [par.rows_per_batch - gamma * par.num_partitions] * par.num_batches
        assignment = assignment.increment(rows, cols, data)
        print(assignment)
        self.assertTrue(assignment.is_valid())
        self.assertEqual(assignment.evaluate(0, 0), 9)
        self.assertEqual(assignment.bound(), 906)
        self.assertEqual(assignment.score, 906)
        return

    def get_parameters(self):
        '''Get some test parameters.'''
        return model.SystemParameters(2, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)

    def get_parameters_2(self):
        '''Get some test parameters.'''
        return model.SystemParameters(6, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)
