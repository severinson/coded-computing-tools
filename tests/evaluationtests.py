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

    def test_heuristic(self):
        '''Test the heuristic solver'''

        with tempfile.TemporaryDirectory() as tmpdirname:
            solver = heuristicsolver.HeuristicSolver()
            simulator = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                             directory=tmpdirname, num_samples=100)
            correct_results = [{'servers': 6, 'batches': 48, 'load': 9000, 'delay': 3.78},
                               {'servers': 6, 'batches': 48, 'load': 9000, 'delay': 3.78},
                               {'servers': 6, 'batches': 48, 'load': 11535, 'delay': 3.9}]

            for par, correct_result in zip(self.get_parameters_partitioning(),
                                           correct_results):

                # Test the solver
                assignment = solver.solve(par)
                self.assertTrue(assignment.is_valid())

                # Test the sampled evaluation
                result = sampled.evaluate(par, assignment, num_samples=100)
                self.assertAlmostEqual(result['servers'], correct_result['servers'],
                                       places=None, delta=correct_result['servers'] * 0.1)
                self.assertAlmostEqual(result['batches'], correct_result['batches'],
                                       places=None, delta=correct_result['batches'] * 0.1)
                self.assertAlmostEqual(result['delay'], correct_result['delay'],
                                       places=None, delta=correct_result['delay'] * 0.1)
                self.assertAlmostEqual(result['load'], correct_result['load'],
                                       places=None, delta=correct_result['load'] * 0.1)

                result = binsearch.evaluate(par, assignment, num_samples=100)
                self.assertAlmostEqual(result['servers'], correct_result['servers'],
                                       places=None, delta=correct_result['servers'] * 0.1)
                self.assertAlmostEqual(result['batches'], correct_result['batches'],
                                       places=None, delta=correct_result['batches'] * 0.1)
                self.assertAlmostEqual(result['delay'], correct_result['delay'],
                                       places=None, delta=correct_result['delay'] * 0.1)
                # self.assertAlmostEqual(result['load'], correct_result['load'],
                #                        places=None, delta=correct_result['load'] * 0.1)

                result = simulator.simulate(par)
                self.assertAlmostEqual(result['servers'][0], correct_result['servers'],
                                       places=None, delta=correct_result['servers'] * 0.1)
                self.assertAlmostEqual(result['batches'][0], correct_result['batches'],
                                       places=None, delta=correct_result['batches'] * 0.1)
                self.assertAlmostEqual(result['delay'][0], correct_result['delay'],
                                       places=None, delta=correct_result['delay'] * 0.1)
                #self.assertAlmostEqual(result['load'][0], correct_result['load'],
                #                       places=None, delta=correct_result['load'] * 0.1)

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
