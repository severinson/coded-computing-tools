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

"""
This module contains unit tests for the various solvers we have developed for
the integer programming problem presented in the paper.

Run the file to run the unit tests.
"""

import unittest
import model
import evaluation
from solvers import randomsolver
from solvers import heuristicsolver
from solvers import greedysolver
from solvers import hybridsolver

class Tests(unittest.TestCase):
    """ Elementary unit tests. """

    def test_random_solver(self):
        """ Some tests on the random assignment solver. """
        par = self.get_parameters()
        solver = randomsolver.RandomSolver()
        assignment = solver.solve(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_greedy_solver(self):
        """ Some tests on the greedy assignment solver. """
        par = self.get_parameters()
        solver = greedysolver.GreedySolver()
        assignment = solver.solve(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_hybrid_assignment(self):
        """ Some tests on the hybrid assignment solver. """
        par = self.get_parameters()
        solver = hybridsolver.HybridSolver(initialsolver=greedysolver.GreedySolver())
        assignment = solver.solve(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_heuristic_solver(self):
        """ Some tests on the heuristic assignment solver. """
        par = self.get_parameters()
        solver = heuristicsolver.HeuristicSolver()
        assignment = solver.solve(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix, verbose=True))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_load_known(self):
        """ Check the load evaluation against the known correct value. """
        known_unicasts = 16
        par = model.SystemParameters(2, # Rows per batch
                                     6, # Number of servers (K)
                                     4, # Servers to wait for (q)
                                     4, # Outputs (N)
                                     1/2, # Server storage (\mu)
                                     1) # Partitions (T)


        solver = randomsolver.RandomSolver()
        assignment = solver.solve(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        solver = greedysolver.GreedySolver()
        assignment = solver.solve(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        solver = hybridsolver.HybridSolver(initialsolver=greedysolver.GreedySolver())
        assignment = solver.solve(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        solver = heuristicsolver.HeuristicSolver()
        assignment = solver.solve(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        return

    def test_delay_known(self):
        """ Check the load evaluation against the known correct value. """
        par = model.SystemParameters(2, # Rows per batch
                                     6, # Number of servers (K)
                                     4, # Servers to wait for (q)
                                     4, # Outputs (N)
                                     1/2, # Server storage (\mu)
                                     1) # Partitions (T)

        solver = randomsolver.RandomSolver()
        assignment = solver.solve(par)
        self.delay_known_comparison(par, assignment)

        solver = greedysolver.GreedySolver()
        assignment = solver.solve(par)
        self.delay_known_comparison(par, assignment)

        solver = hybridsolver.HybridSolver(initialsolver=greedysolver.GreedySolver())
        assignment = solver.solve(par)
        self.delay_known_comparison(par, assignment)

        solver = heuristicsolver.HeuristicSolver()
        assignment = solver.solve(par)
        self.delay_known_comparison(par, assignment)

        return

    def load_eval_comp(self, par, assignment):
        """ Compare the results of the exhaustive and sampled load evaluation. """
        num_samples = 100
        load_sampled = evaluation.communication_load_sampled(par,
                                                             assignment.assignment_matrix,
                                                             assignment.labels,
                                                             num_samples)

        load_exhaustive_avg, load_exhaustive_worst = evaluation.communication_load(par,
                                                                                   assignment.assignment_matrix,
                                                                                   assignment.labels)
        load_difference = abs(load_sampled - load_exhaustive_avg)
        acceptable_ratio = 0.1
        self.assertTrue(load_difference / load_exhaustive_avg < acceptable_ratio)
        self.assertTrue(load_sampled <= load_exhaustive_worst)
        return

    def load_known_comparison(self, par, assignment, known_unicasts):
        """ Check the load evaluation against the known correct value. """
        num_samples = 100
        load_sampled = evaluation.communication_load_sampled(par,
                                                             assignment.assignment_matrix,
                                                             assignment.labels,
                                                             num_samples)

        self.assertEqual(load_sampled, known_unicasts)
        load_exhaustive_avg, load_exhaustive_worst = evaluation.communication_load(par,
                                                                                   assignment.assignment_matrix,
                                                                                   assignment.labels)
        self.assertEqual(load_exhaustive_avg, known_unicasts)
        self.assertEqual(load_exhaustive_worst, known_unicasts)
        return

    def delay_eval_comp(self, par, assignment):
        """ Compare the results of the exhaustive and sampled delay evaluation. """
        num_samples = 100
        delay_sampled = evaluation.computational_delay_sampled(par,
                                                               assignment.assignment_matrix,
                                                               assignment.labels,
                                                               num_samples)

        delay_exhaustive_avg, delay_exhaustive_worst = evaluation.computational_delay(par,
                                                                                      assignment.assignment_matrix,
                                                                                      assignment.labels)

        delay_difference = abs(delay_sampled - delay_exhaustive_avg)
        acceptable_ratio = 0.1
        self.assertTrue(delay_difference / delay_exhaustive_avg < acceptable_ratio)
        self.assertTrue(delay_sampled <= delay_exhaustive_worst)
        return

    def delay_known_comparison(self, par, assignment):
        """ Check the delay evaluation against the known correct value. """
        known_delay = par.computational_delay()
        num_samples = 100
        delay_sampled = evaluation.computational_delay_sampled(par,
                                                               assignment.assignment_matrix,
                                                               assignment.labels,
                                                               num_samples)

        difference = abs(delay_sampled - known_delay)
        self.assertTrue(difference < known_delay / 100)

        delay_exhaustive_avg, delay_exhaustive_worst = evaluation.computational_delay(par,
                                                                                      assignment.assignment_matrix,
                                                                                      assignment.labels)

        difference = abs(delay_exhaustive_worst - known_delay)
        self.assertTrue(difference < known_delay / 100)

        difference = abs(delay_exhaustive_avg - known_delay)
        self.assertTrue(difference < known_delay / 100)

        return

    def get_parameters(self):
        """ Get some test parameters. """
        return model.SystemParameters(2, # Rows per batch
                                      6, # Number of servers (K)
                                      4, # Servers to wait for (q)
                                      4, # Outputs (N)
                                      1/2, # Server storage (\mu)
                                      5) # Partitions (T)

if __name__ == '__main__':
    unittest.main()
