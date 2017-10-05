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

'''This module contains tests for the solvers.

'''

import math
import unittest
import tempfile
import logging
import model
import examples
from solvers import heuristicsolver
from solvers.randomsolver import RandomSolver
from solvers.hybrid import HybridSolver
from assignments.sparse import SparseAssignment
from assignments.cached import CachedAssignment

class HybridSolverTests(unittest.TestCase):
    '''Tests of the hybrid solver.'''

    def test_hybrid(self):
        '''Test the hybrid solver.'''
        logging.basicConfig(level=logging.DEBUG)
        solver = HybridSolver(initialsolver=heuristicsolver.HeuristicSolver())
        parameters = examples.get_parameters_partitioning()[1]
        parameters = self.get_parameters()
        assignment = solver.solve(parameters)
        self.assertTrue(assignment.is_valid())
        return

    def test_deassign_branch_and_bound(self):
        '''Test the solver de-assignment and branch-and-bound.'''
        initialsolver = RandomSolver()
        hybrid = HybridSolver(initialsolver=initialsolver)
        parameters = self.get_parameters()
        assignment = initialsolver.solve(parameters,
                                         assignment_type=CachedAssignment)

        # De-assign a few elements
        partition_count = [0] * parameters.num_partitions
        decrement = 3
        new_assignment = hybrid.deassign(parameters, assignment, partition_count, decrement)

        # Verify that de-assignment worked
        self.assertFalse(new_assignment.is_valid())
        self.assertEqual(sum(partition_count), decrement)
        difference = assignment.assignment_matrix - new_assignment.assignment_matrix
        self.assertEqual(difference.sum(), decrement)

        # Re-assign optimally
        bb_assignment = hybrid.branch_and_bound(parameters, new_assignment, partition_count, assignment)
        self.assertTrue(bb_assignment.is_valid())
        self.assertLessEqual(bb_assignment.score, assignment.score)
        return

    def get_parameters(self):
        '''Get some test parameters.'''
        return model.SystemParameters(6, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      10) # Partitions (T)

class SolverTests(unittest.TestCase):
    '''Tests of the other solvers.'''

    def test_heuristic(self):
        '''Test the heuristic solver'''
        solver = heuristicsolver.HeuristicSolver()
        # Using sparse assignments
        for par in self.get_parameters_partitioning():
            assignment = solver.solve(par, assignment_type=SparseAssignment)
            self.assertTrue(assignment.is_valid())

        # Using cached assignments
        for par in self.get_parameters_partitioning()[:1]:
            assignment = solver.solve(par, assignment_type=CachedAssignment)
            self.assertTrue(assignment.is_valid())

        return

    def test_random(self):
        '''Test the randomized solver'''
        solver = RandomSolver()

        # Using sparse assignments
        for par in self.get_parameters_partitioning():
            assignment = solver.solve(par)
            self.assertTrue(assignment.is_valid())

        # Using cached assignments
        for par in self.get_parameters_partitioning()[:1]:
            assignment = solver.solve(par, assignment_type=CachedAssignment)
            self.assertTrue(assignment.is_valid())

        return

    def get_parameters_partitioning(self):
        '''Get a list of parameters.'''

        rows_per_batch = 250
        num_servers = 9
        q = 6
        num_outputs = q
        server_storage = 1/3
        num_partitions = [2, 100, 3000]
        parameters = list()
        for partitions in num_partitions:
            par = model.SystemParameters(rows_per_batch, num_servers, q,
                                         num_outputs, server_storage,
                                         partitions)
            parameters.append(par)

        return parameters

    def get_parameters(self):
        '''Get some test parameters.'''
        return model.SystemParameters(2, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)

    def get_parameters_partitioning_full(self):
        '''Get a list of parameters for the partitioning plot.'''

        rows_per_batch = 250
        num_servers = 9
        q = 6
        num_outputs = q
        server_storage = 1/3
        num_partitions = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30,
                          40, 50, 60, 75, 100, 120, 125, 150, 200, 250,
                          300, 375, 500, 600, 750, 1000, 1500, 3000]

        parameters = list()
        for partitions in num_partitions:
            par = model.SystemParameters(rows_per_batch, num_servers, q,
                                         num_outputs, server_storage,
                                         partitions)
            parameters.append(par)

        return parameters
