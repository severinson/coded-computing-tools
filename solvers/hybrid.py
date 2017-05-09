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

'''Hybrid assignment solver. Quickly finds a candidate solution and
then improves on it iteratively through branch-and-bound search.

'''

import uuid
import random
import logging
import numpy as np
import model
from assignments.cached import CachedAssignment

class Node(object):
    '''Branch and bound node. Contains an assignment and meta data
    required for the search.

    '''
    def __init__(self, parameters, assignment, row, partition_count):
        '''Create a branch-and-bound node. These nodes are created for each
        branch searched.

        Args:

        parameters: Parametrs object.

        assignment: Assignment object.

        row: The row of the assignment matrix to consider next.

        partition_count: Vector of length num_partitions with symbols
        counts by partition.

        '''
        assert isinstance(parameters, model.SystemParameters)
        assert isinstance(assignment, CachedAssignment)
        assert isinstance(row, int) and 0 <= row < parameters.num_batches
        assert isinstance(partition_count, list)

        self.parameters = parameters
        self.assignment = assignment
        self.complete = False

        # Find the next partially assigned row
        while assignment.batch_union({row}).sum() == parameters.rows_per_batch:
            row += 1
            if row >= parameters.num_batches:
                complete = True
                break

        self.row = row
        self.partition_count = partition_count
        return

    def __str__(self):
        return 'Row: {} Complete: {} Score: {}'.format(self.row, self.complete, self.assignment.score)

class HybridSolver(object):
    '''Hybrid assignment solver. Quickly finds a candidate solution and
    then improves on it iteratively through branch-and-bound search.

    '''

    def __init__(self, initialsolver=None, directory=None):
        '''Create a hybrid solver.

        Args:

        initialsolver: The solver used to find the initial assignment.

        directory: Store intermediate assignments in this directory.
        Set to None to not store intermediate assignments.

        '''
        assert initialsolver is not None
        assert isinstance(directory, str) or directory is None
        self.initialsolver = initialsolver
        self.directory = directory
        return

    def branch_and_bound(self, parameters, assignment, partition_count, best_assignment):
        '''Assign any remaining elements optimally.'''
        stack = list()
        stack.append(Node(parameters, assignment, 0, partition_count))
        completed = 0
        pruned = 0
        while stack:
            node = stack.pop()

            # Store completed nodes with a better score
            if node.complete and node.assignment.score >= best_assignment.score:
                logging.debug('Completed assignment with improvement %d.',
                              best_assignment.score - node.assignment.score)
                best_assignment = node.assignment
                completed += 1
                continue

            for partition in range(parameters.num_partitions):
                if not node.partition_count[partition]:
                    continue

                # Only consider zero-valued elements
                if node.assignment.assignment_matrix[node.row, partition]:
                    continue

                logging.debug('Completed %d, Pruned %d, Stack size %d:', completed, pruned, len(stack))


                assignment = node.assignment.increment([node.row], [partition], [1])
                logging.debug('Best: %d. Bound: %d.', best_assignment.score, assignment.bound())
                if assignment.bound() >= best_assignment.score:
                    pruned += 1
                    continue

                partition_count = node.partition_count[:]
                partition_count[partition] -= 1
                stack.append(Node(parameters, assignment, node.row, partition_count))

        return best_assignment

    def deassign(self, parameters, assignment, partition_count, deassignments):
        '''De-assign an element randomly.

        Args:

        parameters: Parametrs object.

        assignment: Assignment object.

        partition_count: Vector of length num_partitions with symbols
        counts by partition.

        deassignments: Number of deassignments to make.

        Returns: The updated assignment.

        '''
        assert isinstance(deassignments, int) and deassignments > 0

        # Cache the row and col indices to decrement.
        indices = dict()

        # Select row, col pairs.
        while deassignments > 0:
            row = random.randint(0, parameters.num_batches - 1)
            col = random.randint(0, parameters.num_partitions - 1)

            # Ensure that there are remaining values to decrement.
            remaining = assignment.assignment_matrix[row, col] - 1
            if (row, col) in indices:
                remaining -= indices[(row, col)]

            if remaining < 0:
                continue

            # Update the count
            deassignments -= 1
            partition_count[col] += 1
            if (row, col) in indices:
                indices[(row, col)] += 1
            else:
                indices[(row, col)] = 1

        keys = indices.keys()
        values = [indices[key] for key in keys]
        rows = [index[0] for index in keys]
        cols = [index[1] for index in keys]
        return assignment.decrement(rows, cols, values)

    def solve(self, parameters, clear=3):
        '''Find an assignment using this solver.

        Args:

        parameters: System parameters

        Returns: The resulting assignment

        '''
        assert isinstance(parameters, model.SystemParameters)

        # Load solution or find one using the initial solver.
        try:
            assignment = CachedAssignment.load(parameters, directory=self.directory)
            logging.debug('Loaded a candidate solution from disk.')
        except FileNotFoundError:
            logging.debug('Finding a candidate solution using solver %s.', self.initialsolver.identifier)
            assignment = self.initialsolver.solve(parameters, assignment_type=CachedAssignment)

        # Ensure there is room for optimization.
        counts = np.zeros(parameters.num_partitions)
        for row in assignment.rows_iterator():
            counts += row

        if counts.sum() < clear:
            logging.debug('Initial solution leaves no room for optimization. Returning.')
            return assignment

        # Make sure the dynamic programming index is built
        if not assignment.index or not assignment.score:
            assignment = CachedAssignment(parameters, gamma=assignment.gamma,
                                          assignment_matrix=assignment.assignment_matrix,
                                          labels=assignment.labels)

        best_assignment = assignment.copy()

        # Iteratively improve the assignment
        improvement = 1
        iterations = 1
        while improvement / iterations > 0.1:
            # Count symbols by partition
            partition_count = [0] * parameters.num_partitions

            # De-assign elements
            assignment = self.deassign(parameters, assignment, partition_count, clear)

            # Re-assign optimally
            assignment = self.branch_and_bound(parameters, assignment,
                                               partition_count, best_assignment)

            iterations += 1
            improvement += best_assignment.score - assignment.score
            best_assignment = assignment.copy()
            logging.debug('Improved %d over %d iterations.', improvement, iterations)

        return best_assignment

    @property
    def identifier(self):
        '''Return a string identifier for this object.'''
        return self.__class__.__name__
