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
This file contains the code relating to the integer programming
formulation presented in the paper by Albin Severinson, Alexandre Graell
i Amat, and Eirik Rosnes. This includes the formulations itself, and
the solvers presented in the article.

Run the file to run the unit tests.
"""

import itertools as it
import random
import uuid
import os
import unittest
import numpy as np
from scipy.misc import comb as nchoosek
import main

class Perspective(object):
    """ Representation of the count vectors from the perspective of a single server

    Depending on which servers first finish the map computation, a different
    set of batches will be involved in the multicast phase.

    Attributes:
    score: The number of additional rows required to decode all partitions.
    count: An array where count[i] is the number of symbols from partition i.
    rows:
    """

    def __init__(self, score, count, rows, perspective_id=None):
        """ Create a new Perspective object

        Args:
        score: The number of additional rows required to decode all partitions.
        count: An array where count[i] is the number of symbols from partition i.
        rows:
        perspective_id: A unique ID for this instance.
        If None, one is generated.
        """
        self.score = score
        self.count = count
        self.rows = rows

        # Generate a new ID if none was provided
        if isinstance(perspective_id, uuid.UUID):
            self.perspective_id = perspective_id
        else:
            self.perspective_id = uuid.uuid4()

    def __str__(self):
        string = 'Score: ' + str(self.score)
        string += ' Count: ' + str(self.count)
        string += ' Rows: ' + str(self.rows)
        return string

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.rows == other.rows
        return False

    def __hash__(self):
        return hash(self.perspective_id)

    def increment(self, col):
        """ Increment the column and update the indices.

        Args:
        col: The column of the assignment matrix to increment.

        Returns:
        A new Perspective instance with the updated values.
        """

        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] = count[col] + 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

    def decrement(self, col):
        """ Decrement the column and update the indices.

        Args:
        col: The column of the assignment matrix to decrement.

        Returns:
        A new Perspective instance with the updated values.
        """
        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[col] -= 1
        score = remaining_unicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, perspective_id=self.perspective_id)

class BatchResult(object):
    """ The results index from the perspective of a row of the assignment matrix """

    def __init__(self, perspectives=None, summary=None):
        if perspectives is None or summary is None:
            self.perspectives = dict()
        else:
            assert isinstance(perspectives, dict)
            assert isinstance(summary, list)
            assert isinstance(summary, list)
            self.perspectives = perspectives
            self.summary = summary

        return

    def __getitem__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        return self.perspectives[key]

    def __setitem__(self, key, value):
        if not isinstance(key, Perspective) or not isinstance(value, Perspective):
            raise TypeError('Key and value must be of type perspective')

        self.perspectives[key] = value

    def __delitem__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        del self.perspectives[key]

    def __contains__(self, key):
        if not isinstance(key, Perspective):
            raise TypeError('Key must be of type perspective')

        return key in self.perspectives


    def init_summary(self, par):
        """ Build a summary of the number of perspectives that need symbols
        from a certain partition.
        """

        num_perspectives = len(self.perspectives)

        # Build a simple summary that keeps track of how many symbols
        # that need a certain partition.
        self.summary = [num_perspectives for x in range(par.num_partitions)]

    def keys(self):
        """ Return the keys of the perspectives dict. """
        return self.perspectives.keys()

    def copy(self):
        """ Returns a shallow copy of itself """
        perspectives = dict(self.perspectives)
        summary = list(self.summary)
        return BatchResult(perspectives, summary)

class Assignment(object):
    """ Storage design representation

    Representing a storage assignment. Has support for dynamic programming
    methods to efficiently find good assignments for small instances.

    Attributes:
    assignment_matrix: An num_batches by num_partitions Numpy array, where
    assignment_matrix[i, j] is the number of rows from partition j stored in
    batch i.
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    score: The sum of all unicasts that need to be sent until all partitions
    can be decoded over all subsets that can complete the map phase. If either
    score or index is set to None, the index will be built from scratch. Set
    both to False to prevent the index from being built.
    index: A list of BatchResult of length equal to the number of batches. If
    either score or index is set to None, the index will be built from scratch.
    Set both to False to prevent the index from being built.
    """

    def __init__(self, par, assignment_matrix=None, labels=None, score=None, index=None):
        self.par = par

        if assignment_matrix is None:
            self.assignment_matrix = np.zeros([par.num_batches, par.num_partitions])
        else:
            self.assignment_matrix = assignment_matrix

        if labels is None:
            self.labels = [set() for x in range(par.num_servers)]
            self.label()
        else:
            self.labels = labels

        if score is None or index is None:
            self.build_index()
        else:
            self.score = score
            self.index = index

    def __str__(self):
        string = ''
        string += 'assignment matrix:\n'
        string += str(self.assignment_matrix) + '\n'
        string += 'labels:\n'
        string += str(self.labels) + '\n'
        string += 'Score: ' + str(self.score)
        return string

    def save(self, directory='./assignments/'):
        """ Save the assignment to disk

        Args:
        directory: Directory to save to
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + self.par.identifier() + '.npy', self.assignment_matrix)
        return

    # TODO: Throw exception if file doesn't exist
    @classmethod
    def load(cls, par, directory='./assignments/'):
        """ Load assignment from disk

        Args:
        par: System parameters
        directory: Directory to load from

        Returns:
        The loaded assignment
        """
        assignment_matrix = np.load(directory + par.identifier() + '.npy')
        return cls(par, assignment_matrix=assignment_matrix, score=False, index=False)

    def build_index(self):
        """ Build the dynamic programming index

        Build an index pairing rows of the assignment matrix to which
        perspectives they appear in. Only run once when creating a new
        assignment.
        """

        self.score = 0

        # Index for which sets every row is contained in
        self.index = [BatchResult(self.par) for x in range(self.par.num_batches)]
        self.num_subsets = nchoosek(self.par.num_servers, self.par.q)
        subsets = it.combinations(range(self.par.num_servers), self.par.q)

        # Build an index for which count vectors every row is part of
        for Q in subsets:
            for k in Q:
                rows = set()
                for batch in self.labels[k]:
                    rows.add(batch)

                for j in range(self.par.sq, int(self.par.server_storage*self.par.q) + 1):
                    for subset in it.combinations([x for x in Q if x != k], j):
                        rows = rows | set.intersection(*[self.labels[x] for x in subset])

                selector_vector = np.zeros(self.par.num_batches)
                for row in rows:
                    selector_vector[row] = 1

                count_vector = np.dot(selector_vector, self.assignment_matrix)
                count_vector -= self.par.num_source_rows / self.par.num_partitions
                score = remaining_unicasts(count_vector)
                self.score = self.score + score

                perspective = Perspective(score, count_vector, rows)
                for row in rows:
                    assert perspective not in self.index[row]
                    self.index[row][perspective] = perspective

        # Initialize the summaries
        for x in self.index:
            x.init_summary(self.par)

    def label(self):
        """ Label the batches with server subsets

        Label all batches with subsets in the order given by itertools.combinations
        """
        assert self.par.server_storage * self.par.q % 1 == 0, 'Must be integer'
        labels = it.combinations(range(self.par.num_servers),
                                 int(self.par.server_storage * self.par.q))
        row = 0
        for label in labels:
            for server in label:
                self.labels[server].add(row)
            row += 1
        return

    def bound(self):
        """ Compute a bound for this assignment """

        assert self.index and self.score, 'Cannot compute bound if there is no index.'
        decreased_unicasts = 0
        for row_index in range(self.assignment_matrix.shape[0]):
            row = self.assignment_matrix[row_index]
            remaining_assignments = self.par.rows_per_batch - sum(row)
            assert remaining_assignments >= 0
            decreased_unicasts += max(self.index[row_index].summary) * remaining_assignments

        # Bound can't be less than 0
        return max(self.score - decreased_unicasts, 0)

    def increment(self, row, col):
        """ Increment assignment_matrix[row, col]

        Increment the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:
        row: The row index
        col: The column index

        Returns:
        A new assignment object.
        """

        assert self.index and self.score, 'Cannot increment if there is no index.'
        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] = assignment_matrix[row, col] + 1
        objective_value = self.score

        # Select the perspectives linked to the row
        perspectives = index[row]

        # Iterate over all perspectives linked to that row
        for perspective_key in perspectives.keys():
            perspective = perspectives[perspective_key]

            # Create a new perspective from the updated values
            new_perspective = perspective.increment(col)

            # Update the objective function
            objective_value = objective_value - (perspective.score - new_perspective.score)

            # Update the index for all rows which include this perspective
            for perspective_row in perspective.rows:
                assert hash(new_perspective) == hash(index[perspective_row][perspective])
                index[perspective_row][perspective] = new_perspective

            # Update the summaries if the count reached zero for the
            # indicated column
            if new_perspective.count[col] == 0:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] = index[perspective_row].summary[col] - 1

        # Return a new assignment object
        return Assignment(self.par,
                          assignment_matrix=assignment_matrix,
                          labels=self.labels,
                          score=objective_value,
                          index=index)

    def decrement(self, row, col):
        """ Decrement assignment_matrix [row, col]

        Decrement the element at [row, col] and update the objective
        value. Returns a new assignment object. Does not change the
        current assignment object.

        Args:
        row: The row index
        col: The column index

        Returns:
        A new assignment object.
        """

        assert self.index and self.score, 'Cannot decrement if there is no index.'
        assert self.assignment_matrix[row, col] >= 1, 'Can\'t decrement a value less than 1.'

        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        assignment_matrix = np.array(self.assignment_matrix)
        assignment_matrix[row, col] = assignment_matrix[row, col] - 1
        objective_value = self.score

        # Select the perspectives linked to the row
        perspectives = index[row]

        # Iterate over all perspectives linked to that row
        for perspective_key in perspectives.keys():
            perspective = perspectives[perspective_key]

            # Create a new perspective from the updated values
            new_perspective = perspective.decrement(col)

            # Update the objective function
            objective_value = objective_value - (perspective.score - new_perspective.score)

            # Update the index for all rows which include this perspective
            for perspective_row in perspective.rows:
                assert hash(new_perspective) == hash(index[perspective_row][perspective])
                index[perspective_row][perspective] = new_perspective

            # Update the summaries if the count reached zero for the
            # indicated column
            if new_perspective.count[col] == -1:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] += 1

        # Return a new assignment object
        return Assignment(self.par,
                          assignment_matrix=assignment_matrix,
                          labels=self.labels,
                          score=objective_value,
                          index=index)

    def evaluate(self, row, col):
        """ Return the performance change that incrementing (row, col)
        would induce without changing the assignment.
        """

        assert self.index and self.score, 'Cannot evaluateif there is no index.'
        return self.index[row].summary[col]

class Node(object):
    """ Branch-and-bound node """

    def __init__(self, par, assignment, row, symbols_separated):
        assert isinstance(assignment, Assignment)
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
        root = Node(par, Assignment(par), 0, symbols_separated)

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

    assignment = Assignment(par)

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

    #[partial_partitions.add(x) for x in range(par.num_partitions)
    #if rows_by_partition[x] < coded_rows_per_partition]

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
            assignment_matrix[row, col] = assignment_matrix[row, col] + 1
            rows_by_partition[col] += 1
            row_sum += 1

            # Remove the partition index if there are no more assignments
            if rows_by_partition[col] == coded_rows_per_partition:
                partial_partitions.remove(col)

def assignment_random(par):
    """ Create an assignment matrix randomly.

    Args:
    par: System parameters

    Returns:
    The resulting assignment object.
    """

    assignment = Assignment(par, score=False, index=False)
    assignment_matrix = assignment.assignment_matrix
    rows_by_partition = [0 for x in range(par.num_partitions)]
    assign_remaining_random(par, assignment_matrix, rows_by_partition)
    return assignment

def communication_load(par, assignment_matrix, labels, Q=None):
    """ Calculate the communication load of an assignment.

    Count the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment exhaustively.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    Q: Leave at None to evaluate all subsets Q, or set to a tuple of servers to
    count unicasts only for some specific Q.

    Returns:
    Tuple with the first element the average number of unicasts required and
    second element the worst case number of unicasts.
    """

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        subsets = [Q]
        num_subsets = 1

    for Q in subsets:
        set_score = 0

        # Count over all server perspectives
        for k in Q:
            # Create the set of all symbols stored at k or sent to k
            # via multicast.
            rows = set()
            #[rows.add(row) for row in labels[k]]
            for batch in labels[k]:
                rows.add(batch)

            for j in range(par.sq, int(par.server_storage*par.q) + 1):
                for subset in it.combinations([x for x in Q if x != k], j):
                    rows = rows | set.intersection(*[labels[x] for x in subset])

            selector_vector = np.zeros(par.num_batches)
            for row in rows:
                selector_vector[row] = 1

            count_vector = np.dot(selector_vector, assignment_matrix)
            count_vector -= par.num_source_rows / par.num_partitions
            score = remaining_unicasts(count_vector)

            total_score = total_score + score
            set_score = set_score + score

        if set_score > worst_score:
            worst_score = set_score

    average_score = total_score / num_subsets
    return average_score, worst_score

def computational_delay(par, assignment_matrix, labels, Q=None):
    """ Calculate the computational delay of an assignment.

    Calculate the number of servers required to decode all partitions for an
    assignment exhaustively.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.

    Returns:
    Number of servers required averaged over all subsets.
    """

    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        subsets = it.combinations(range(par.num_servers), par.q)
        num_subsets = nchoosek(par.num_servers, par.q)
    else:
        subsets = [Q]
        num_subsets = 1

    for Q in subsets:
        set_score = 0

        # Count the total number of symbols per partition
        rows_by_partition = np.zeros(par.num_partitions)

        # Keep track of the batches we've added
        batches_in_Q = set()

        # Add the batches from the first q servers
        for server in Q:
            for batch in labels[server]:
                batches_in_Q.add(batch)

        for batch in batches_in_Q:
            rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])
        set_score = par.q

        # TODO: Add more servers until all partitions can be decoded
        if not enough_symbols(par, rows_by_partition):
            set_score += 1

        total_score += set_score
        if set_score > worst_score:
            worst_score = set_score

    average_score = total_score / num_subsets
    return average_score, worst_score

def enough_symbols(par, rows_by_partition):
    """" Return True if there are enough symbols to decode all partitions.

    Args:
    par: System parameters object
    rows_by_partition: A numpy array of row counts by partition.

    Returns:
    True if there are enough symbols to decode all partitions. Otherwise false.
    """

    assert len(rows_by_partition) == par.num_partitions, \
        'The input array must have length equal to the number of partitions.'

    for num_rows in rows_by_partition:
        if num_rows < par.rows_per_partition:
            return False

    return True

def computational_delay_sampled(par, assignment_matrix, labels, num_samples):
    """ Estimate the computational delay of an assignment.

    Estimate the number of servers required to decode all partitions for an
    assignment through Monte Carlo simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    num_runs: Number of runs

    Returns:
    Number of servers required averaged over all n runs.
    """
    total_score = 0

    for _ in range(num_samples):

        # Generate a random sequence of servers
        Q = random.sample(range(par.num_servers), par.num_servers)

        # Count the total number of symbols per partition
        rows_by_partition = np.zeros(par.num_partitions)

        # Keep track of the batches we've added
        batches_added = set()

        # Add the batches from the first q servers
        #[[batches_added.add(x) for x in labels[y]] for y in Q[0:par.q]]
        for server in Q[0:par.q]:
            for batch in labels[server]:
                batches_added.add(batch)

        for batch in batches_added:
            rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])
        total_score = total_score + par.q

        # Add batches from more servers until there are enough rows to
        # decode all partitions.
        for server in Q[par.q:]:
            if enough_symbols(par, rows_by_partition):
                break

            # Add the rows from the batches not already counted
            for batch in {x for x in labels[server] if x not in batches_added}:
                rows_by_partition = np.add(rows_by_partition, assignment_matrix[batch])

            # Keep track of which batches we've added
            #[batches_added.add(x) for x in labels[server]]
            for batch in labels[server]:
                batches_added.add(batch)

            # Update the score
            total_score = total_score + 1

    average_score = total_score / num_samples
    return average_score

def remaining_unicasts(rows_by_partition):
    """" Return the number of unicasts required to decode all partitions.

    Args:
    rows_by_partition: A numpy array of row counts by partition.

    Returns:
    The number of unicasts required to decode all partitions.
    """

    unicasts = 0
    for num_rows in rows_by_partition:
        if num_rows < 0:
            unicasts = unicasts - num_rows

    return unicasts

def communication_load_sampled(par, assignment_matrix, labels, num_samples):
    """ Estimate the communication load of an assignment.

    Estimate the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment through Monte Carlo
    simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the
    rows of the assignment_matrix stored at server i.
    num_runs: Number of runs

    Returns:
    Number of unicasts required averaged over all n runs.
    """

    total_score = 0
    server_list = list(range(par.num_servers))

    for _ in range(num_samples):

        # Generate a random Q and k
        Q = random.sample(server_list, par.q)
        k = random.sample(Q, 1)[0]

        # Sum the corresponding rows of the assignment matrix
        batches = set()
        for batch in labels[k]:
            batches.add(batch)

        # Add the rows sent in the shuffle phase
        for j in range(par.sq, int(par.server_storage*par.q) + 1):
            for subset in it.combinations([x for x in Q if x != k], j):
                batches = batches | set.intersection(*[labels[x] for x in subset])

        count_vector = np.zeros(par.num_partitions) - par.num_source_rows / par.num_partitions
        for batch in batches:
            count_vector = np.add(count_vector, assignment_matrix[batch])

        # Calculate the score
        score = remaining_unicasts(count_vector)
        total_score = total_score + score

    average_score = total_score * par.q / num_samples
    return average_score

def objective_function_sampled(par, assignment_matrix, labels, num_samples=1000):
    """ Return the estimated communication load and computational delay.

    Estimate the number of unicasts required for all servers to hold enough
    symbols to deocde all partitions for some assignment through Monte Carlo
    simulations.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    labels: List of sets, where the elements of the set labels[i] are the rows
    of the assignment_matrix stored at server i.
    n: Number of runs

    Returns:
    Tuple with first element estimated number of unicasts, and second estimated
    number of servers required to wait for.
    """
    delay = computational_delay_sampled(par, assignment_matrix, labels, num_samples)
    load = communication_load_sampled(par, assignment_matrix, labels, num_samples)
    return load, delay

def is_valid(par, assignment_matrix, verbose=False):
    """ Evaluate if an assignment is valid and complete.

    Args:
    par: System parameters
    assignment_matrix: Assignment matrix
    verbose: Print why an assignment might be invalid

    Returns:
    True if the assignment matrix is valid and complete. False otherwise.
    """
    for row in assignment_matrix:
        if row.sum() != par.rows_per_batch:
            if verbose:
                print('Row', row, 'does not sum to the number of rows per batch. Is:',
                      row.sum(), 'Should be:', par.rows_per_batch)
            return False

    for col in assignment_matrix.T:
        if col.sum() != par.num_coded_rows / par.num_partitions:
            if verbose:
                print('Column', col, 'does not sum to the number of rows per partition. Is',
                      col.sum(), 'Should be:', par.num_coded_rows / par.num_partitions)
            return False

    return True

class Tests(unittest.TestCase):
    """ Elementary unit tests. """

    def test_is_valid(self):
        """ Verify that is_valid works correctly. """
        par = self.get_parameters()
        assignment_matrix = np.array([[2, 0, 0, 0, 0],
                                      [0, 2, 0, 0, 0],
                                      [0, 0, 2, 0, 0],
                                      [0, 0, 0, 2, 0],
                                      [0, 0, 0, 0, 2],
                                      [2, 0, 0, 0, 0],
                                      [0, 2, 0, 0, 0],
                                      [0, 0, 2, 0, 0],
                                      [0, 0, 0, 2, 0],
                                      [0, 0, 0, 0, 2],
                                      [2, 0, 0, 0, 0],
                                      [0, 2, 0, 0, 0],
                                      [0, 0, 2, 0, 0],
                                      [0, 0, 0, 2, 0],
                                      [0, 0, 0, 0, 2]])

        self.assertTrue(is_valid(par, assignment_matrix))
        assignment_matrix[0, 0] = 0
        self.assertFalse(is_valid(par, assignment_matrix))
        return

    def test_labels(self):
        """ Verify that the labeling is valid. """
        par = self.get_parameters()
        assignment = Assignment(par, score=False, index=False)
        batches_per_server = par.server_storage * par.num_source_rows / par.rows_per_batch
        for batches in assignment.labels:
            self.assertEqual(len(batches), batches_per_server)
        return

    def test_random_assignment(self):
        """ Some tests on the random assignment solver. """
        par = self.get_parameters()
        assignment = assignment_random(par)
        self.assertTrue(is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_greedy_assignment(self):
        """ Some tests on the greedy assignment solver. """
        par = self.get_parameters()
        assignment = assignment_greedy(par)
        self.assertTrue(is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_hybrid_assignment(self):
        """ Some tests on the hybrid assignment solver. """
        par = self.get_parameters()
        assignment = assignment_hybrid(par)
        self.assertTrue(is_valid(par, assignment.assignment_matrix))
        self.load_eval_comp(par, assignment)
        self.delay_eval_comp(par, assignment)
        return

    def test_load_known(self):
        """ Check the load evaluation against the known correct value. """
        known_unicasts = 16
        par = main.SystemParameters(2, # Rows per batch
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
        return

    def load_eval_comp(self, par, assignment):
        """ Compare the results of the exhaustive and sampled load evaluation. """
        num_samples = 100
        load_sampled = communication_load_sampled(par,
                                                  assignment.assignment_matrix,
                                                  assignment.labels,
                                                  num_samples)

        load_exhaustive_avg, load_exhaustive_worst = communication_load(par,
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
        load_sampled = communication_load_sampled(par,
                                                  assignment.assignment_matrix,
                                                  assignment.labels,
                                                  num_samples)

        self.assertEqual(load_sampled, known_unicasts)
        load_exhaustive_avg, load_exhaustive_worst = communication_load(par,
                                                                        assignment.assignment_matrix,
                                                                        assignment.labels)
        self.assertEqual(load_exhaustive_avg, known_unicasts)
        self.assertEqual(load_exhaustive_worst, known_unicasts)
        return

    def delay_eval_comp(self, par, assignment):
        """ Compare the results of the exhaustive and sampled delay evaluation. """
        num_samples = 100
        delay_sampled = computational_delay_sampled(par,
                                                    assignment.assignment_matrix,
                                                    assignment.labels,
                                                    num_samples)

        delay_exhaustive_avg, delay_exhaustive_worst = computational_delay(par,
                                                                           assignment.assignment_matrix,
                                                                           assignment.labels)

        delay_difference = abs(delay_sampled - delay_exhaustive_avg)
        acceptable_ratio = 0.1
        self.assertTrue(delay_difference / delay_exhaustive_avg < acceptable_ratio)
        self.assertTrue(delay_sampled <= delay_exhaustive_worst)
        return

    def get_parameters(self):
        """ Get some test parameters. """
        return main.SystemParameters(2, # Rows per batch
                                     6, # Number of servers (K)
                                     4, # Servers to wait for (q)
                                     4, # Outputs (N)
                                     1/2, # Server storage (\mu)
                                     5) # Partitions (T)
if __name__ == '__main__':
    unittest.main()
