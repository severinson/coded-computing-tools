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
from solvers import Solver
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
        self.row = row

        # Find the next partially assigned row
        while (self.row < parameters.num_batches and
               assignment.batch_union({self.row}).sum() == parameters.rows_per_batch):
            self.row += 1

        # Mark assignment as completed if no partial rows
        if self.row == parameters.num_batches:
            self.complete = True
            self.row = None

        self.partition_count = partition_count
        return

    def __str__(self):
        return 'Row: {} Complete: {} Score: {}'.format(self.row, self.complete, self.assignment.score)

class HybridSolver(Solver):
    '''Hybrid assignment solver. Quickly finds a candidate solution and
    then improves on it iteratively through branch-and-bound search.

    '''

    def __init__(self, initialsolver=None, directory=None, clear=3):
        '''Create a hybrid solver.

        Args:

        initialsolver: The solver used to find the initial assignment.

        directory: Store intermediate assignments in this directory.
        Set to None to not store intermediate assignments.

        clear: Number of elements of the assignment matrix to
        re-assign per iteration.

        '''
        assert initialsolver is not None
        assert isinstance(directory, str) or directory is None
        self.initialsolver = initialsolver
        self.directory = directory
        self.clear = clear
        return

    def branch_and_bound(self, parameters, assignment, partition_count, best_assignment):
        '''Assign any remaining elements optimally.'''
        stack = list()
        stack.append(Node(parameters, assignment, 0, partition_count))
        completed = 0
        pruned = 0
        best_assignment = best_assignment
        while stack:
            node = stack.pop()

            # Store completed nodes with a better score
            if node.complete and node.assignment.score <= best_assignment.score:
                logging.debug('Completed assignment with improvement %d.',
                              best_assignment.score - node.assignment.score)
                best_assignment = node.assignment
                completed += 1

            for partition in range(parameters.num_partitions):
                if not node.partition_count[partition]:
                    continue

                # Only consider zero-valued elements
                # if node.assignment.assignment_matrix[node.row, partition]:
                #     continue

                # logging.debug('Completed %d, Pruned %d, Stack size %d: row/col: [%d, %d]',
                #               completed, pruned, len(stack), node.row, partition)

                assignment = node.assignment.increment([node.row], [partition], [1])
                # logging.debug('Best: %d. Bound: %d.', best_assignment.score, assignment.bound())
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

    def solve(self, parameters, assignment_type=None):
        '''Find an assignment using this solver.

        Args:

        parameters: System parameters

        Returns: The resulting assignment

        '''
        assert isinstance(parameters, model.SystemParameters)
        assert assignment_type is None or assignment_type is CachedAssignment, \
            'Solver must be used with CachedAssignment.'

        # Load solution or find one using the initial solver.
        try:
            assignment = CachedAssignment.load(parameters, directory=self.directory)
            logging.debug('Loaded a candidate solution from disk.')
        except FileNotFoundError:
            logging.debug('Finding a candidate solution using solver %s.', self.initialsolver.identifier)
            assignment = self.initialsolver.solve(parameters, assignment_type=CachedAssignment)
            if self.directory:
                assignment.save(directory=self.directory)

        # Ensure there is room for optimization.
        counts = np.zeros(parameters.num_partitions)
        for row in assignment.rows_iterator():
            counts += row

        if counts.sum() < self.clear:
            logging.debug('Initial solution leaves no room for optimization. Returning.')
            return assignment

        # Make sure the dynamic programming index is built
        if not assignment.index or not assignment.score:
            assignment = CachedAssignment(parameters, gamma=assignment.gamma,
                                          assignment_matrix=assignment.assignment_matrix,
                                          labels=assignment.labels)

        original_score = max(assignment.score, 1)
        best_assignment = assignment.copy()

        # Iteratively improve the assignment
        iterations = 0
        total_improvement = 0
        moving_average = 1
        stop_threshold = 0.0001
        while moving_average > stop_threshold:
            # Count symbols by partition
            partition_count = [0] * parameters.num_partitions

            # De-assign elements
            decremented_assignment = self.deassign(parameters, best_assignment,
                                                   partition_count, self.clear)

            # Re-assign optimally
            improved_assignment = self.branch_and_bound(parameters, decremented_assignment,
                                                        partition_count, best_assignment)

            iterations += 1
            improvement = (best_assignment.score - improved_assignment.score) / original_score
            total_improvement += improvement
            moving_average *= 0.9
            moving_average += improvement
            best_assignment = improved_assignment.copy()
            logging.info('Improved %f%% over %d iterations. Moving average: %f%%. Stop threshold: %f%%.',
                         total_improvement * 100, iterations, moving_average * 100, stop_threshold * 100)

            if self.directory and improvement > 0:
                best_assignment.save(directory=self.directory)

        return best_assignment

    @property
    def identifier(self):
        '''Return a string identifier for this object.'''
        return self.__class__.__name__
