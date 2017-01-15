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
This module contains the various solvers we have developed for the integer
programming problem presented in the paper.

Run the file to run the unit tests.
"""

import itertools as it
import random
import uuid
import unittest
import numpy as np
from scipy.misc import comb as nchoosek
import model
import evaluation

class Node(object):
    """ Branch-and-bound node """

    def __init__(self, par, assignment, row, symbols_separated):
        assert isinstance(assignment, model.Assignment)
        assert isinstance(row, int)
        assert isinstance(symbols_separated, list)

        self.par = par

        self.assignment = assignment

        # Current row being assigned
        self.row = row

        self.symbols_separated = symbols_separated

        self.node_id = uuid.uuid4()
        return

    def __hash__(self):
        return hash(self.node_id)

    def remaining(self):
        """ Return the number of remaining assignments for the current row. """
        return self.par.rows_per_batch - self.assignment.assignment_matrix[self.row].sum()

def branch_and_bound(par, score=None, root=None, verbose=False):
    """ Branch-and-bound assignment solver

    Finds an assignment through branch-and-bound search.

    Args:
    par: System parameters
    score: Score of the current node.
    root: Node to start searching from.
    verbose: Print extra messages if True

    Returns:
    The resulting assignment
    """

    # Initialize the solver with a greedy solution
    if score is None:
        best_assignment = assignment_greedy(par)
        score = best_assignment.score
    else:
        best_assignment = None
        score = score

    if verbose:
        print('Starting B&B with score:', score)

    # Set of remaining nodes to explore
    remaining_nodes = list()

    # Separate symbols by partition
    symbols_per_partition = par.num_coded_rows / par.num_partitions
    symbols_separated = [symbols_per_partition for x in range(par.num_partitions)]

    # Add the root of the tree
    if root is None:
        root = Node(par, model.Assignment(par), 0, symbols_separated)

    remaining_nodes.append(root)

    searched = 0
    pruned = 0

    while len(remaining_nodes) > 0:
        node = remaining_nodes.pop()
        searched = searched + 1

        # While there are no more valid assignments for this row, move
        # to the next one.
        while node.remaining() == 0:
            node.row = node.row + 1
            if node.row == par.num_batches:
                break

        # If there are no more assignments for this node, we've found
        # a solution.
        if node.row == par.num_batches:
            #print('Found a solution:', node.assignment)
            #print('---------------------')

            # If this solution is better than the previous best, store it
            if node.assignment.score < score:
                best_assignment = node.assignment
                score = best_assignment.score

            continue

        # Iterate over all possible branches from this node
        for col in range(par.num_partitions):

            # Continue if there are no more valid assignments for this
            # column/partition
            if node.symbols_separated[col] == 0:
                continue

            updated_assignment = node.assignment.increment(node.row, col)

            # Continue if the bound for this assignment is no better
            # than the previous best solution.
            if updated_assignment.bound() >= score:
                pruned = pruned + 1
                continue

            row = node.row

            # Decrement the count of remaining symbols for the
            # assigned partition.
            updated_symbols_separated = list(node.symbols_separated)
            updated_symbols_separated[col] = updated_symbols_separated[col] - 1

            if verbose:
                print('Bound:', updated_assignment.bound(),
                      'Row:', row,
                      '#nodes:', len(remaining_nodes),
                      "Best:", score,
                      '#searched:', searched, '#pruned:', pruned)

            # Add the new node to the set of nodes to be explored
            remaining_nodes.append(Node(par, updated_assignment, row, updated_symbols_separated))

    return best_assignment

def assignment_greedy(par, verbose=False):
    """ Greedy assignment solver.

    Finds an assignment by assigning one row at a time, always greedily
    selecting the row with largest improvement.

    Args:
    par: System parameters
    verbose: Print extra messages if True

    Returns:
    The resulting assignment
    """

    assignment = model.Assignment(par)

    # Separate symbols by partition
    symbols_per_partition = par.num_coded_rows / par.num_partitions
    symbols_separated = [symbols_per_partition for x in range(par.num_partitions)]

    # Assign symbols row by row
    for row in range(par.num_batches):
        if verbose:
            print('Assigning row', row, 'Bound:', assignment.bound())

        # Assign rows_per_batch rows per batch
        for _ in range(par.rows_per_batch):
            score_best = 0
            best_col = 0

            # Try one column at a time
            for col in range(par.num_partitions):

                # If there are no more symbols left for this column to
                # assign, continue
                if symbols_separated[col] == 0:
                    continue

                # Evaluate the score of the assignment
                score_updated = assignment.evaluate(row, col)
                score_updated = score_updated + symbols_separated[col] / symbols_per_partition

                # If it's better, store it
                if score_updated > score_best:
                    score_best = score_updated
                    best_col = col

            # Store the best assignment from this iteration
            assignment = assignment.increment(row, best_col)
            symbols_separated[best_col] = symbols_separated[best_col] - 1

    return assignment

def assignment_hybrid(par, clear=10, min_runs=200, verbose=False):
    """ Hybrid assignment solver.

    Quickly finds a candidate solution and then improves it iteratively.

    Args:
    par: System parameters
    clear: Max assignment matrix elements to decrement per iteration.
    min_runs: The solver runs at least min_runs/c iterations, where c is the
    number of dercemented elements.
    verbose: Print extra messages if True

    Returns:
    The resulting assignment
    """

    # Start off with a greedy assignment
    if verbose:
        print('Finding a candidate solution using the greedy solver.')

    assignment = assignment_greedy(par, verbose=verbose)
    best_assignment = assignment

    # Then iteratively try to improve it by de-assigning a random set
    # of batches and re-assigning them using the optimal
    # branch-and-bound solver.
    for c in range(3, clear + 1):
        if verbose:
            print('Improving it by de-assigning', c, 'random symbols at a time.')

        improvement = min_runs/c #  To make sure it runs at least this many times
        iterations = 1
        while improvement / iterations >= 1:
            score = best_assignment.score
            assignment = best_assignment

            # Keep track of the number of remaining symbols per partition
            symbols_separated = [0 for x in range(par.num_partitions)]

            # Clear a few symbols randomly
            for _ in range(c):
                row = random.randint(0, par.num_batches - 1)
                col = random.randint(0, par.num_partitions - 1)

                # While the corresponding element is zero, randomize
                # new indices.
                while assignment.assignment_matrix[row, col] <= 0:
                    row = random.randint(0, par.num_batches - 1)
                    col = random.randint(0, par.num_partitions - 1)

                assignment = assignment.decrement(row, col)
                symbols_separated[col] = symbols_separated[col] + 1

            # Apply the branch-and-bound solver
            root = Node(par, assignment, 0, symbols_separated)
            new_assignment = branch_and_bound(par, score=score, root=root, verbose=verbose)

            # If it found a better solution, overwrite the current one
            if new_assignment is not None:
                best_assignment = new_assignment
                improvement = improvement + score - best_assignment.score
                if verbose:
                    print('Iteration finished with an improvement of',
                          score - best_assignment.score)

            iterations = iterations + 1

    return best_assignment

def assign_block(par, assignment_matrix, row, min_col, max_col):
    """ Assign rows to all slots between min_col and max_col.

    This method won't work when max_col - min_col > par.rows_per_batch.

    Args:
    par: System parameters
    assignment_matrix: The assignment matrix
    row: The assignment matrix rows to assign rows to
    min_col: Start of block
    max_col: End of block

    Returns:
    The wrapped column index. If there are 10 columns and max_col is 12, 2 is returned.
    """

    assert (max_col - min_col) <= par.rows_per_batch

    if min_col == max_col:
        return max_col

    wrapped_col = max_col
    if max_col >= par.num_partitions:
        wrapped_col = max_col - par.num_partitions
        assignment_matrix[row, 0:wrapped_col] += 1

    max_col = min(par.num_partitions, max_col)
    assignment_matrix[row][min_col:max_col] += 1

    return wrapped_col

def assignment_heuristic(par, verbose=True):
    """ Create an assignment using a block-diagonal structure.

    Args:
    par: System parametetrs
    verbose: Print extra messages if True.

    Returns: The resulting assignment
    """

    assignment = model.Assignment(par, score=False, index=False)
    assignment_matrix = assignment.assignment_matrix

    # If there are more coded rows than matrix elements, add that
    # number of rows to each element.
    rows_per_element = int(par.num_coded_rows / (par.num_partitions * par.num_batches))
    rows_by_partition = [par.num_batches * rows_per_element] * par.num_partitions
    if rows_per_element > 0:
        for row in range(par.num_batches):
            for col in range(par.num_partitions):
                assignment_matrix[row, col] = rows_per_element

    # Assign the remaining rows in a block-diagonal fashion
    remaining_rows_per_batch = par.rows_per_batch - rows_per_element * par.num_partitions
    min_col = 0
    for row in range(par.num_batches):
        max_col = min_col + remaining_rows_per_batch
        min_col = assign_block(par, assignment_matrix, row, min_col, max_col)

    return assignment

    coded_rows_per_partition = par.num_coded_rows / par.num_partitions
    partial_partitions = set()

    for partition in range(par.num_partitions):
        if rows_by_partition[partition] < coded_rows_per_partition:
            partial_partitions.add(partition)

    # Assign any remaining rows randomly
    # Assign symbols randomly row-wise
    for row in range(par.num_batches):
        row_sum = assignment_matrix[row].sum()
        incremented_partitions = set()

        while row_sum < par.rows_per_batch:

            # Get a random column index corresponding to a partial partition
            partial_non_incremented = {col for col in partial_partitions
                                       if col not in incremented_partitions}

            if len(partial_non_incremented) > 0:
                col = random.sample(partial_non_incremented, 1)[0]
            else:
                col = random.sample(partial_partitions, 1)[0]

            #print(assignment_matrix[row], partial_non_incremented, col)
            """
            while assignment_matrix[row, col] > rows_per_element and len(partial_partitions) > 1:
                print(row, col, partial_partitions)
                col = random.sample(partial_partitions, 1)[0]
            """

            # Increment the count
            assignment_matrix[row, col] += 1
            rows_by_partition[col] += 1
            incremented_partitions.add(col)
            row_sum += 1

            # Remove the partition index if there are no more assignments
            if rows_by_partition[col] == coded_rows_per_partition:
                partial_partitions.remove(col)

    return assignment

def assignment_annealing(par, num_samples=100, verbose=False):
    """ Create an assignment through simulated annealing.

    A candidate solution is generated randomly. Parts of the rows are
    de-assigned and re-assigned randomly. If the new solution performs better,
    it replaces the old. Solutions are evaluated through sampling, so the
    current solution will sometimes be replaced by a worse solution.

    Args:
    par: System parametetrs
    num_samples: Number of samples taken when evaluating a solution.
    verbose: Extra printouts if True.

    Returns:
    The resulting assignment.
    """

    # Find a candidate solution
    if verbose:
        print('Finding a candidate solution.')

    num_subsets = nchoosek(par.num_servers, par.q)
    assignment = assignment_random(par)
    best_assignment = assignment
    load = evaluation.communication_load_sampled(par, best_assignment.assignment_matrix, best_assignment.labels, num_samples)
    load /= num_subsets * par.q
    delay = evaluation.computational_delay_sampled(par, best_assignment.assignment_matrix, best_assignment.labels, num_samples)
    best_score = load + delay

    num_clear = max(int(par.num_coded_rows / 10), 3)
    coded_rows_per_partition = par.num_coded_rows / par.num_partitions

    while num_clear > 3:
        if verbose:
            print('De-assigning', num_clear, 'rows per iteration.')

        improvement = 100
        num_iterations = 1

        # Clear and re-assign rows until the average improvement drops
        # below the threshold.
        while improvement / num_iterations > 0.01 or num_iterations < par.num_coded_rows / num_clear:
            assignment = best_assignment.copy()

            # Clear rows
            rows_by_partition = [coded_rows_per_partition] * par.num_partitions
            for _ in range(num_clear):
                row = random.randint(0, par.num_batches - 1)
                col = random.randint(0, par.num_partitions - 1)

                # While the corresponding element is zero, randomize
                # new indices.
                while assignment.assignment_matrix[row, col] <= 0:
                    row = random.randint(0, par.num_batches - 1)
                    col = random.randint(0, par.num_partitions - 1)

                assignment.assignment_matrix[row, col] -= 1
                rows_by_partition[col] -= 1

            # Re-assign the rows
            assign_remaining_random(par, assignment.assignment_matrix, rows_by_partition)
            load = evaluation.communication_load_sampled(par, assignment.assignment_matrix, assignment.labels, num_samples)
            load /= num_subsets * par.q
            delay = evaluation.computational_delay_sampled(par, assignment.assignment_matrix, assignment.labels, num_samples)
            new_score = load + delay

            print('Best score:', best_score, 'New score:', new_score)
            if new_score < best_score:
                improvement += best_score - new_score
                if verbose:
                    print('Iteration finished with an improvement of',
                          best_score - new_score, 'Score:', new_score)

                best_assignment = assignment
                best_score = new_score

            num_iterations += 1

        num_clear = int(num_clear / 2)

    return best_assignment

def assign_remaining_random(par, assignment_matrix, rows_by_partition):
    """ Assign any remaining rows randomly.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    rows_by_partition: A list of row counts by partition.
    """

    assert len(rows_by_partition) == par.num_partitions, \
        'rows_by_partition must be of length equal to the number of partitions.'

    # Create a set containing the indices of the partially assigned
    # partitions
    coded_rows_per_partition = par.num_coded_rows / par.num_partitions
    partial_partitions = set()

    for partition in range(par.num_partitions):
        if rows_by_partition[partition] < coded_rows_per_partition:
            partial_partitions.add(partition)

    # Assign symbols randomly row-wise
    for row in range(par.num_batches):
        row_sum = assignment_matrix[row].sum()
        while row_sum < par.rows_per_batch:

            # Get a random column index corresponding to a partial partition
            col = random.sample(partial_partitions, 1)[0]

            # Increment the count
            assignment_matrix[row, col] += 1
            rows_by_partition[col] += 1
            row_sum += 1

            # Remove the partition index if there are no more assignments
            if rows_by_partition[col] == coded_rows_per_partition:
                partial_partitions.remove(col)

def assignment_random(par, verbose=False):
    """ Create an assignment matrix randomly.

    Args:
    par: System parameters
    verbose: Unused

    Returns:
    The resulting assignment object.
    """

    assignment = model.Assignment(par, score=False, index=False)
    assignment_matrix = assignment.assignment_matrix
    rows_by_partition = [0 for x in range(par.num_partitions)]
    assign_remaining_random(par, assignment_matrix, rows_by_partition)
    return assignment

class Tests(unittest.TestCase):
    """ Elementary unit tests. """

    def test_random_assignment(self):
        """ Some tests on the random assignment solver. """
        par = self.get_parameters()
        assignment = assignment_random(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_greedy_assignment(self):
        """ Some tests on the greedy assignment solver. """
        par = self.get_parameters()
        assignment = assignment_greedy(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_hybrid_assignment(self):
        """ Some tests on the hybrid assignment solver. """
        par = self.get_parameters()
        assignment = assignment_hybrid(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_annealing_assignment(self):
        """ Some tests on the hybrid assignment solver. """
        par = self.get_parameters()
        assignment = assignment_annealing(par)
        self.assertTrue(model.is_valid(par, assignment.assignment_matrix, verbose=True))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_heuristic_assignment(self):
        """ Some tests on the heuristic assignment solver. """
        par = self.get_parameters()
        assignment = assignment_heuristic(par)
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

        assignment = assignment_random(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        assignment = assignment_greedy(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        assignment = assignment_hybrid(par)
        self.load_known_comparison(par, assignment, known_unicasts)

        assignment = assignment_annealing(par)
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

        assignment = assignment_random(par)
        self.delay_known_comparison(par, assignment)

        assignment = assignment_greedy(par)
        self.delay_known_comparison(par, assignment)

        assignment = assignment_hybrid(par)
        self.delay_known_comparison(par, assignment)

        assignment = assignment_annealing(par)
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
