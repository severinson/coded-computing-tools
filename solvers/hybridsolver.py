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

""" Hybrid assignment solver.
Quickly finds a candidate solution and then improves it iteratively through
branch-and-bound search.
"""

import uuid
import random
import model

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

class HybridSolver(object):
    """ Hybrid assignment solver.
    Quickly finds a candidate solution and then improves it iteratively through
    branch-and-bound search.
    """

    def __init__(self, initialsolver=None, directory=None, clear=10, min_runs=200):
        """ Create a hybrid solver.

        Args:
        initialsolver: The solver to use when finding an initial assignment.
        directory: If not none, store intermediate assignments in this directory.
        clear: Max assignment matrix elements to decrement per iteration.
        min_runs: The solver runs at least min_runs/c iterations, where c is the
        number of dercemented elements.
        """

        assert initialsolver is not None
        assert directory is None or isinstance(directory, str)

        self.initialsolver = initialsolver
        self.directory = directory
        self.clear = clear
        self.min_runs = min_runs
        return

    def tree_search(self, par, score, remaining_nodes, verbose=False):
        """ Perform the tree search.

        Args:
        par: System parameters
        score: Score of the initial assignment.
        remaining_nodes: List of the remaining nodes to search.
        verbose: Print extra messages if True

        Returns:
        The resulting assignment
        """

        best_assignment = None
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
                remaining_nodes.append(Node(par, updated_assignment, row,
                                            updated_symbols_separated))

        return best_assignment

    def branch_and_bound(self, par, score=None, root=None, verbose=False):
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

        assert score is not None

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
        best_assignment = self.tree_search(par, score, remaining_nodes, verbose=verbose)
        return best_assignment

    def solve(self, par, verbose=False):
        """ Hybrid assignment solver.

        Quickly finds a candidate solution and then improves it iteratively.

        Args:
        par: System parameters
        verbose: Print extra messages if True

        Returns:
        The resulting assignment
        """

        assert isinstance(par, model.SystemParameters)

        # If there already is a solution on disk, load it
        try:
            assignment = model.Assignment.load(par, directory=self.directory)

            if verbose:
                print('Loaded a candidate solution from disk.')

        # Otherwise find one using a solver
        except FileNotFoundError:
            if verbose:
                print('Finding a candidate solution using', self.initialsolver.identifier)

            assignment = self.initialsolver.solve(par, verbose=verbose)

        # Build the dynamic programming index if we didn't do so already.
        if not assignment.index:
            assignment = model.Assignment(par,
                                          assignment_matrix=assignment.assignment_matrix,
                                          labels=assignment.labels)

        best_assignment = assignment

        # Then iteratively try to improve it by de-assigning a random set
        # of batches and re-assigning them using the optimal
        # branch-and-bound solver.
        for num_clear in range(3, self.clear + 1):
            if verbose:
                print('Improving it by de-assigning', num_clear, 'random symbols at a time.')

            improvement = self.min_runs/num_clear #  To make sure it runs at least this many times
            iterations = 1
            while improvement / iterations >= 1:
                score = best_assignment.score
                assignment = best_assignment

                # Keep track of the number of remaining symbols per partition
                symbols_separated = [0 for x in range(par.num_partitions)]

                # Clear a few symbols randomly
                for _ in range(num_clear):
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
                new_assignment = self.branch_and_bound(par, score=score, root=root, verbose=verbose)

                # If it found a better solution, overwrite the current one
                if new_assignment is not None:
                    best_assignment = new_assignment
                    improvement = improvement + score - best_assignment.score
                    if verbose:
                        print('Iteration finished with an improvement of',
                              score - best_assignment.score)

                    # Save the best assignment so far.
                    if self.directory is not None:
                        if verbose:
                            print('Saving assignment to disk.')

                        best_assignment.save(directory=self.directory)

                iterations = iterations + 1

        return best_assignment

    @property
    def identifier(self):
        """ Return a string identifier for this object. """
        return self.__class__.__name__
