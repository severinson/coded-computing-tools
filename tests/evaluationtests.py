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

'''This module contains tests for the evaluation, and the simulation.

'''

import unittest
import tempfile
import model
import simulation
from solvers import heuristicsolver
from solvers import randomsolver
from evaluation import sampled
from evaluation import binsearch
from evaluation import analytic

class EvaluationTests(unittest.TestCase):
    '''Tests for the evaluation'''

    def test_lt(self):
        '''Test the analytic LT code evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.lt_performance(par)
            self.verify_result(result, correct_result, delta=0.01)

        return

    def test_mds(self):
        '''Test the analytic MDS code evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.mds_performance(par)
            self.verify_result(result, correct_result, delta=0.01)

            # self.assertAlmostEqual(result['servers'], correct_result['servers'],
            #                        places=None, delta=correct_result['servers'] * 0.1)
            # self.assertAlmostEqual(result['batches'], correct_result['batches'],
            #                        places=None, delta=correct_result['batches'] * 0.1)

        return


    def verify_result(self, result, correct_result, delta=0.2):
        '''Check the results against known correct results.

        Args:

        result: Measured result.

        correct_result: Dict with correct results.

        delta: Correct result must be within a delta fraction of the
        measured result.

        '''
        for key, value in correct_result.items():
            self.assertAlmostEqual(result[key].mean(), value, places=None, delta=value * delta)

    def verify_solver(self, solvefun, correct_results, delta=0.2):
        '''Check the results from evaluating the assignment produced by some
        solver against known correct results.

        Args:

        solvefun: Function called to create an assignment.

        correct_results: List of dicts with correct results.

        delta: Correct result must be within a delta fraction of the
        measured result.

        '''
        for par, correct_result in zip(self.get_parameters_partitioning(), correct_results):
            assignment = solvefun(par)
            self.assertTrue(assignment.is_valid())

            result = binsearch.evaluate(par, assignment, num_samples=100)
            self.verify_result(result, correct_result)

        return

    def test_heuristic(self):
        '''Test the heuristic solver'''
        solver = heuristicsolver.HeuristicSolver()
        correct_results = [{'servers': 6, 'batches': 48, 'unicasts_strat_1': 9000, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'unicasts_strat_1': 9000, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'unicasts_strat_1': 11535, 'delay': 12}]

        self.verify_solver(solver.solve, correct_results)
        return

    def test_heuristic_analytic(self):
        '''Test the analytic heuristic assignment evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'delay': 11.3}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                        correct_results):
             result = analytic.average_heuristic(par)
             self.verify_result(result, correct_result, delta=0.1)

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
