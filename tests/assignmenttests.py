import unittest
import tempfile
import numpy as np
import model

class ModelTests(unittest.TestCase):
    '''Tests for the model module.'''

    def test_is_valid(self):
        '''Verify that is_valid works correctly.'''
        par = self.get_parameters()
        assignment_matrix = np.array([[2, 0, 0, 0, 0],
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

        self.assertTrue(model.is_valid(par, assignment_matrix))
        assignment_matrix[0, 0] = 0
        self.assertFalse(model.is_valid(par, assignment_matrix))
        return

    def test_labels(self):
        '''Verify that the labeling is valid.'''
        par = self.get_parameters()
        assignment = model.Assignment(par, score=False, index=False)
        batches_per_server = par.server_storage * par.num_source_rows / par.rows_per_batch
        for batches in assignment.labels:
            self.assertEqual(len(batches), batches_per_server)
        return

    def test_save_load(self):
        '''Verify that saving and loading works.'''
        par = self.get_parameters()
        assignment = model.Assignment(par)
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
            assignment_loaded = model.Assignment.load(par, directory=tmpdirname)
            self.assertTrue(np.array_equal(assignment.assignment_matrix, assignment_loaded.assignment_matrix))

        return

    def test_load(self):
        '''Verify that loading non-existent files throws the correct
        exception.

        '''
        par = self.get_parameters()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(FileNotFoundError):
                assignment_loaded = model.Assignment.load(par, directory=tmpdirname)
        return

    def get_parameters(self):
        '''Get some test parameters.'''
        return model.SystemParameters(2, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)
