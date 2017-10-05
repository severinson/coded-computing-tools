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

import math
import unittest
import tempfile
import model
import simulation
from solvers.heuristicsolver import HeuristicSolver
from solvers import randomsolver
from evaluation import sampled
from evaluation import binsearch
from evaluation import analytic

class EvaluationTests(unittest.TestCase):
    '''Tests for the evaluation'''

    def test_lt(self):
        '''Test the analytic LT code evaluation.'''
        correct_results = [{'servers': 7, 'delay': 13.97},
                           {'servers': 7, 'delay': 13.97},
                           {'servers': 7, 'delay': 13.97}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.lt_performance(par)
            self.verify_result(result, correct_result, delta=0.01)

        return

    def test_mds(self):
        '''Test the analytic MDS code evaluation.'''
        correct_results = [{'servers': 6, 'batches': 48, 'delay': 11.97},
                           {'servers': 6, 'batches': 48, 'delay': 11.97},
                           {'servers': 6, 'batches': 48, 'delay': 11.97}]

        for par, correct_result in zip(self.get_parameters_partitioning(),
                                       correct_results):
            result = analytic.mds_performance(par)
            self.verify_result(result, correct_result, delta=0.01)

            # self.assertAlmostEqual(result['servers'], correct_result['servers'],
            #                        places=None, delta=correct_result['servers'] * 0.1)
            # self.assertAlmostEqual(result['batches'], correct_result['batches'],
            #                        places=None, delta=correct_result['batches'] * 0.1)

        return


    def verify_result(self, result, correct_result, delta=0.1):
        '''Check the results against known correct results.

        Args:

        result: Measured result.

        correct_result: Dict with correct results.

        delta: Correct result must be within a delta fraction of the
        measured result.

        '''
        for key, value in correct_result.items():
            if value == math.inf:
                self.assertAlmostEqual(result[key].mean(), value, places=1,
                                       msg='key={}, value={}'.format(str(key), str(value)))
            else:
                self.assertAlmostEqual(result[key].mean(), value, delta=value*delta,
                                       msg='key={}, value={}'.format(str(key), str(value)))

    def verify_solver(self, solver, parameters, correct_results, delta=0.2):
        '''Check the results from evaluating the assignment produced by some
        solver against known correct results.

        Args:

        solver: Assignment solver.

        parameters: System parameters.

        correct_results: List of dicts with correct results.

        delta: Correct result must be within a delta fraction of the
        measured result.

        '''
        evaluator = binsearch.SampleEvaluator(num_samples=1000)
        for par, correct_result in zip(parameters, correct_results):
            assignment = solver.solve(par)
            self.assertTrue(assignment.is_valid())

            result = evaluator.evaluate(par, assignment)
            self.verify_result(result, correct_result)

        return

    def test_heuristic(self):
        '''Test the heuristic solver'''
        solver = HeuristicSolver()
        correct_results = [{'servers': 6, 'batches': 48, 'unicast_load_1': 1.5, 'delay': 11.3},
                           {'servers': 6, 'batches': 48, 'unicast_load_1': 1.5, 'delay': 11.3},
                           {'servers': 6, 'batches': 49.512, 'unicast_load_1': 1.836, 'delay': 11.56}]

        parameters = self.get_parameters_partitioning()
        self.verify_solver(solver, parameters, correct_results)
        return

    def test_evaluation_1(self):
        '''Test the evaluation.'''
        parameters = model.SystemParameters(rows_per_batch=2, num_servers=6, q=4, num_outputs=4,
                                            server_storage=1/2, num_partitions=1)
        correct = {'servers': 4, 'batches': 20, 'delay': 7.1333333333333293,
                   'unicast_load_1': 0.8, 'multicast_load_1': 0.6,
                   'unicast_load_2': 0.8, 'multicast_load_2': math.inf}
        solver = HeuristicSolver()
        self.verify_solver(solver, [parameters], [correct])
        return

    def test_evaluation_2(self):
        '''Test the evaluation.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=10, q=9, num_outputs=9,
                                            server_storage=1/3, num_partitions=5)
        correct = {'servers': 9, 'batches': 324, 'delay': 25.460714285714285,
                   'unicast_load_1': 720 / 540, 'multicast_load_1': 840 / 540,
                   'unicast_load_2': 0, 'multicast_load_2': 1470 / 540}
        solver = HeuristicSolver()
        self.verify_solver(solver, [parameters], [correct])
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
