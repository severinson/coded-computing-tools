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
from solvers import treesolver
from solvers.hybrid import HybridSolver
from evaluation import sampled
from assignments.sparse import SparseAssignment
from assignments.cached import CachedAssignment

class TreeSolverTests(object):
    '''Tests for the tree solver.'''
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

    def test_weight_distribution_1(self):
        '''Test the distribution of tree weights used by the tree solver.'''
        distribution = treesolver.weight_distribution(10, 2, 1)
        self.assertEqual(sum(distribution), 1)
        return

    def test_weight_distribution_2(self):
        '''Test the distribution of tree weights used by the tree solver.'''
        distribution = treesolver.weight_distribution(100, 20, 30)
        self.assertAlmostEqual(sum(distribution), 30)
        return

    def test_weight_distribution_3(self):
        '''Test the distribution of tree weights used by the tree solver.'''
        distribution = treesolver.weight_distribution(100, 20, 0)
        self.assertAlmostEqual(sum(distribution), 0)
        return

    def test_weight_distribution_4(self):
        '''Test the distribution of tree weights used by the tree solver.'''
        distribution = treesolver.weight_distribution(3000, 100, 200)
        self.assertAlmostEqual(sum(distribution), 200)
        return

    def test_distribution_truncation_1(self):
        '''Test the distribution truncation used by the tree solver.'''
        distribution = [1,2,3,4,5,4,3,2,1,0]
        limits = [4] * len(distribution)
        truncated_distribution = treesolver.truncate_distribution(distribution, limits,
                                                                   sum(distribution))
        self.assertEqual(truncated_distribution[-1], 0)
        self.assertEqual(truncated_distribution[0], truncated_distribution[-2])
        self.assertAlmostEqual(sum(distribution), sum(truncated_distribution))
        for value, limit in zip(truncated_distribution, limits):
            self.assertLessEqual(value, limit)
        return

    def test_distribution_truncation_2(self):
        '''Test the distribution truncation used by the tree solver.'''
        distribution = list(range(100))
        limits = [50] * len(distribution)
        truncated_distribution = treesolver.truncate_distribution(distribution, limits,
                                                                   sum(distribution))
        self.assertEqual(truncated_distribution[0], 0)
        self.assertAlmostEqual(sum(distribution), sum(truncated_distribution))
        for value, limit in zip(truncated_distribution, limits):
            self.assertLessEqual(round(value, 7), limit)
        return

    def test_distribution_truncation_3(self):
        '''Test the distribution truncation used by the tree solver.'''
        distribution = [1, 0, 0, 0]
        limits = [1, 3, 7, 9]
        truncated_distribution = treesolver.truncate_distribution(distribution, limits, 1)
        self.assertAlmostEqual(sum(truncated_distribution), 1)
        for value, limit in zip(truncated_distribution, limits):
            self.assertLessEqual(round(value, 7), limit)
        return

    def test_distribution_truncation_4(self):
        '''Test the distribution truncation used by the tree solver.'''
        distribution = list(range(3000))
        limits = [1501] * len(distribution)
        truncated_distribution = treesolver.truncate_distribution(distribution, limits,
                                                                   sum(distribution))
        self.assertEqual(truncated_distribution[0], 0)
        self.assertAlmostEqual(sum(distribution), sum(truncated_distribution))
        for value, limit in zip(truncated_distribution, limits):
            self.assertLessEqual(round(value, 7), limit)
        return

    def test_integer_distribution_1(self):
        '''Test the distribution of integers used by the tree solver.'''
        limits = [5] * 20
        distribution = treesolver.integer_distribution(20, 3, 10, limits=limits)
        self.assertEqual(sum([x[1] for x in distribution]), 10)
        for i, value in distribution:
            self.assertLessEqual(value, limits[i])
        return

    def test_integer_distribution_2(self):
        '''Test the distribution of integers used by the tree solver.'''
        limits = list(range(50)) + list(range(49, -1, -1))
        distribution = treesolver.integer_distribution(100, 3, 20, limits=limits)
        self.assertEqual(sum([x[1] for x in distribution]), 20)
        for i, value in distribution:
            self.assertLessEqual(value, limits[i])
        return

    def test_integer_distribution_3(self):
        '''Test the distribution of integers used by the tree solver.'''
        limits = [math.inf] * 100
        distribution = treesolver.integer_distribution(100, 3, 0, limits=limits)
        self.assertEqual(len(distribution), 0)
        return

    def test_integer_distribution_4(self):
        '''Test the distribution of integers used by the tree solver.'''
        limits = [1501] * 3000
        distribution = treesolver.integer_distribution(3000, 100, 200, limits=limits)
        self.assertEqual(sum([x[1] for x in distribution]), 200)
        for i, value in distribution:
            self.assertLessEqual(value, limits[i])
        return

    def assert_prefix(self, prefix, pre_fork_remaining_values, max_index, correct_num_children):
        '''Make assertitions checking a batch prefix.'''
        assert isinstance(prefix, treesolver.BatchPrefix)
        num_children = 0
        child_indices = set()

        self.assertEqual(prefix.max_index, max_index)
        if len(prefix.indices):
            self.assertEqual(prefix.indices[-1], max_index)

        self.assertEqual(prefix.children, correct_num_children)
        for partition, children in prefix.distribution:
            self.assertEqual(children + prefix.remaining_values[partition],
                             pre_fork_remaining_values[partition])
            self.assertFalse(partition in child_indices)
            child_indices.add(partition)
            num_children += children

        return

    def assert_child(self, parent, child, pre_fork_remaining_values):
        '''Make assertitions checking a batch prefix in relation to its parent.'''
        if parent.remaining:
            self.assertEqual(child.remaining, parent.remaining - 1)
            self.assertEqual(len(child.indices), len(parent.indices) + 1)
        else:
            self.assertEqual(child.remaining, 0)
            self.assertEqual(child.children, parent.children - 1)
            self.assertEqual(len(child.indices), len(parent.indices))

        self.assertEqual(child.partitions, parent.partitions)

        if child.indices and not child.remaining:
            self.assertEqual(child.remaining_values[child.max_index],
                             pre_fork_remaining_values[child.max_index])
        elif child.indices:
            self.assertEqual(child.remaining_values[child.max_index],
                             pre_fork_remaining_values[child.max_index])

        return

    def test_batch_prefix_create(self):
        '''Test the BatchPrefix class.'''
        indices = []
        remaining = 10
        partitions = 40
        num_children = 36
        remaining_values = [9] * partitions
        pre_fork_remaining_values = remaining_values[:]
        prefix = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)

        self.assertEqual(prefix.remaining, remaining)
        self.assertEqual(prefix.partitions, partitions)
        self.assertEqual(prefix.children, num_children)
        self.assertEqual(prefix.max_index, -1)
        return

    def test_batch_prefix_fork_1(self):
        '''Test the BatchPrefix class.'''
        indices = []
        remaining = 10
        partitions = 40
        num_children = 36
        remaining_values = [9] * partitions
        parent = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)

        pre_fork_remaining_values = parent.remaining_values[:]
        child, parent = parent.fork()
        while child:
            self.assert_child(parent, child, pre_fork_remaining_values)
            pre_fork_remaining_values = parent.remaining_values[:]
            child, parent = parent.fork()

        return

    def test_batch_prefix_fork_2(self):
        '''Test the BatchPrefix class.'''
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        remaining = 1
        partitions = 40
        num_children = 1
        remaining_values = [8] * 9 + [9] * (partitions - 9)
        parent = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)

        pre_fork_remaining_values = parent.remaining_values[:]
        child, parent = parent.fork()
        self.assert_child(parent, child, pre_fork_remaining_values)
        self.assertEqual(child.fork(), (None, None))
        self.assertEqual(parent.fork(), (None, None))
        return

    def test_batch_prefix_fork_3(self):
        '''Test the BatchPrefix class.'''
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        remaining = 0
        partitions = 40
        num_children = 2
        remaining_values = [8] * 10 + [9] * (partitions - 10)
        parent = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)
        self.assertEqual(parent.children, 1)

        pre_fork_remaining_values = parent.remaining_values[:]
        child, parent = parent.fork()
        self.assertTrue(parent is None)
        self.assertEqual(child.children, 0)
        return

    def test_batch_prefix_fork_4(self):
        '''Test the BatchPrefix class.'''

        indices = []
        remaining = 10
        partitions = 40
        num_children = 36
        remaining_values = [9] * partitions
        parent = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)
        stack = [parent]
        completed = 0
        while stack:
            parent = stack.pop()
            pre_fork_remaining_values = parent.remaining_values[:]
            parent_children = parent.children
            child, parent = parent.fork()
            # print('stack size:', len(stack))
            # print('CHILD:', child)
            # print('PARENT:', parent)
            # print('completed', completed, 'batches.')

            if parent:
                stack.append(parent)

            if child:
                if not child.remaining:
                    self.assertEqual(len(child.indices), remaining)
                    completed += 1
                stack.append(child)

            if parent and child:
                self.assert_child(parent, child, pre_fork_remaining_values)

        self.assertEqual(sum(remaining_values), 0)
        self.assertEqual(completed, num_children)
        return

    def test_batch_prefix_fork_5(self):
        '''Test the BatchPrefix class.'''
        return
        indices = []
        remaining = 100
        partitions = 150
        num_children = 36
        remaining_values = [24] * partitions
        parent = treesolver.BatchPrefix(indices, remaining, partitions,
                                         num_children, remaining_values)
        stack = [parent]
        completed = 0
        while stack:
            parent = stack.pop()
            pre_fork_remaining_values = parent.remaining_values[:]
            parent_children = parent.children
            child, parent = parent.fork()
            # print('stack size:', len(stack))
            # print('CHILD:', child)
            # print('PARENT:', parent)
            # print('completed', completed, 'batches.')

            if parent:
                stack.append(parent)

            if child:
                if not child.remaining:
                    self.assertEqual(len(child.indices), remaining)
                    completed += 1
                stack.append(child)

            if parent and child:
                self.assert_child(parent, child, pre_fork_remaining_values)

        self.assertEqual(sum(remaining_values), 0)
        self.assertEqual(completed, num_children)
        return

    def test_tree(self):
        '''Test the complete tree solver.'''
        solver = treesolver.TreeSolver()
        for parameters in self.get_parameters_partitioning()[0:2]:
            assignment = solver.solve(parameters)
            self.assertTrue(assignment.is_valid())

        return

    def test_tree_distribution(self):
        '''Test the distribution of assignments over batches used by the tree
        solver.

        '''
        return
        for parameters in self.get_parameters_partitioning():
            rows_per_element = parameters.num_coded_rows
            rows_per_element /= parameters.num_partitions * parameters.num_batches
            rows_per_element = math.floor(rows_per_element)

            remaining_rows_per_batch = parameters.rows_per_batch
            remaining_rows_per_batch -= rows_per_element * parameters.num_partitions
            remaining_rows_per_batch = round(remaining_rows_per_batch)
            if remaining_rows_per_batch == 0:
                continue

            remaining_rows_per_partition = parameters.num_coded_rows / parameters.num_partitions
            remaining_rows_per_partition -= rows_per_element * parameters.num_batches
            remaining_rows_per_partition = round(remaining_rows_per_partition)

            distribution = treesolver.assignment_distribution(parameters.num_partitions,
                                                               remaining_rows_per_batch,
                                                               parameters.num_batches)

            # print(distribution)
            assignments = [x[1] for x in distribution]
            self.assertEqual(sum(assignments), parameters.num_batches)
            self.assertTrue(max(assignments) <= remaining_rows_per_partition,
                            str(max(assignments)) + ' ' + str(remaining_rows_per_partition) + '\n' + str(distribution))
            self.assertTrue(min(assignments) >= 0)

            # Check that last value is valid
            # partition_index, assignment = distribution[0]
            # print(distribution[0], assignment, parameters.num_partitions)
            # self.assertGreater(parameters.num_partitions - partition_index, assignment)

        return

    def test_tree_batch_generator(self):
        '''Test the batches created for the tree solver.

        '''
        return
        parameters = self.get_parameters_partitioning_full()[13]
        rows_per_element = parameters.num_coded_rows
        rows_per_element /= parameters.num_partitions * parameters.num_batches
        rows_per_element = math.floor(rows_per_element)

        remaining_rows_per_batch = parameters.rows_per_batch
        remaining_rows_per_batch -= rows_per_element * parameters.num_partitions
        remaining_rows_per_batch = round(remaining_rows_per_batch)

        num_batches = 0
        for batch in treesolver.batch_generator(parameters):
            print('Tested', num_batches, 'batches.')
            print('-----------------------------------------------------------')
            self.assertEqual(len(batch), remaining_rows_per_batch)
            num_batches += 1

        self.assertEqual(num_batches, parameters.num_batches)
        return

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
