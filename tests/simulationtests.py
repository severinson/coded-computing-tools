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

'''This module contains tests of the simulation module.

'''

import os
import math
import unittest
import tempfile
import pandas as pd
import simulation

from functools import partial
from model import SystemParameters
from solvers.heuristicsolver import HeuristicSolver
from evaluation.binsearch import SampleEvaluator

class EvaluationTests(unittest.TestCase):
    '''Tests of the simulation module.'''

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

    def verify_solver(self, solver, parameters, correct_results):
        '''Check the results from evaluating the assignment produced by some
        solver against known correct results.

        Args:

        solver: Assignment solver.

        parameters: System parameters.

        correct_results: List of dicts with correct results.

        '''
        evaluator = binsearch.SampleEvaluator(num_samples=1000)
        for par, correct_result in zip(parameters, correct_results):
            assignment = solver.solve(par)
            self.assertTrue(assignment.is_valid())

            result = evaluator.evaluate(par, assignment)
            self.verify_result(result, correct_result)

        return

    def test_simulation(self):
        '''Test basic functionality.'''
        parameters = SystemParameters(rows_per_batch=5, num_servers=10, q=9, num_outputs=9,
                                      server_storage=1/3, num_partitions=5)
        correct = {'servers': 9, 'batches': 324, 'delay': 25.460714285714285/9,
                   'unicast_load_1': 720/540/9, 'multicast_load_1': 840/540/9,
                   'unicast_load_2': 0, 'multicast_load_2': 1470/540/9}
        solver = HeuristicSolver()
        evaluator = SampleEvaluator(num_samples=1000)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, parameters.identifier() + '.csv')
            dataframe = simulation.simulate(
                parameters,
                directory=tmpdir,
                rerun=False,
                samples=10,
                solver=solver,
                assignment_eval=evaluator,
            )
            self.verify_result(dataframe, correct)

            simulate_fun = partial(
                simulation.simulate,
                directory=tmpdir,
                rerun=False,
                samples=10,
                solver=solver,
                assignment_eval=evaluator,
            )
            dataframe = simulation.simulate_parameter_list(
                parameter_list=[parameters],
                simulate_fun=simulate_fun,
                map_complexity_fun=lambda x: 1,
                encode_delay_fun=lambda x: 0,
                reduce_delay_fun=lambda x: 0,
            )
            self.verify_result(dataframe, correct)

        return
