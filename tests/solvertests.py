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

'''This module contains tests for the solvers, the evaluation, and the
simulation.

'''

import unittest
import tempfile
import model
import simulation
from solvers import heuristicsolver
from solvers import randomsolver
from evaluation import sampled
from evaluation import analytic

class SolverTests(unittest.TestCase):
    '''Tests for the solvers'''

    def test_lt(self):
        '''Test the analytic LT code evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.9}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.lt_performance(par)
            self.assertAlmostEqual(result['servers'], correct_result['servers'],
                                   places=None, delta=correct_result['servers'] * 0.1)
            self.assertAlmostEqual(result['batches'], correct_result['batches'],
                                   places=None, delta=correct_result['batches'] * 0.1)

        return

    def test_mds(self):
        '''Test the analytic MDS code evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.9}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.mds_performance(par)
            self.assertAlmostEqual(result['servers'], correct_result['servers'],
                                   places=None, delta=correct_result['servers'] * 0.1)
            self.assertAlmostEqual(result['batches'], correct_result['batches'],
                                   places=None, delta=correct_result['batches'] * 0.1)

        return

    def test_heuristic_analytic(self):
        '''Test the analytic heuristic assignment evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.78},
                           {'servers': 6, 'batches': 48, 'delay': 3.9}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.average_heuristic(par)
            self.assertAlmostEqual(result['servers'], correct_result['servers'],
                                   places=None, delta=correct_result['servers'] * 0.1)
            self.assertAlmostEqual(result['batches'], correct_result['batches'],
                                   places=None, delta=correct_result['batches'] * 0.1)

        return

    def test_heuristic(self):
        '''Test the heuristic solver'''
        solver = heuristicsolver.HeuristicSolver()
        for par in self.get_parameters_partitioning():
            assignment = solver.solve(par)
            self.assertTrue(assignment.is_valid())

        return

    def test_random(self):
        '''Test the randomized solver'''
        solver = randomsolver.RandomSolver()
        for par in self.get_parameters_partitioning():
            assignment = solver.solve(par)
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
