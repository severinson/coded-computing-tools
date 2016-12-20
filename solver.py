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

# This file contains the code relating to the integer programming
# formulation used in the paper by Albin Severinson, Alexandre Graell
# i Amat, and Eirik Rosnes. This includes the formulations itself, and
# the solvers presented in the article.

import numpy as np
from scipy.misc import comb as nchoosek
import itertools as it
import random
import math
import uuid
import multiprocessing as mp
import asyncio

# Representation of the count vectors from the perspective of a single server
class Perspective(object):
    def __init__(self, score, count, rows, id=None):
        self.score = score
        self.count = count
        self.rows = rows

        # Generate a new ID if none was provided
        if type(id) is uuid.UUID:
            self.id = id
        else:
            self.id = uuid.uuid4()

    def __str__(self):
        s = 'Score: ' + str(self.score)
        s = s + ' Count: ' + str(self.count)
        s = s + ' Rows: ' + str(self.rows)
        return s

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.rows == other.rows
        return False

    def __hash__(self):
        return hash(self.id)

    # Increment the indicated column
    def increment(self, col):

        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[0][col] = count[0][col] + 1
        score = remainingUnicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, id=self.id)

    # Decrement the indicated column
    def decrement(self, col):
        # Copy the count array
        count = np.array(self.count)

        # Increment the indicated column and calculate the updated score
        count[0][col] = count[0][col] - 1
        score = remainingUnicasts(count)

        # Return a new instance with the updated values
        return Perspective(score, count, self.rows, id=self.id)

# The results index from the perspective of a row of the assignment matrix
class BatchResult(object):
    def __init__(self, perspectives=None, summary=None, extended_summary=None):
        if perspectives is None or summary is None or extended_summary is None:
            self.perspectives = dict()
        else:
            assert type(perspectives) is dict
            assert type(summary) is list
            assert type(summary) is list
            self.perspectives = perspectives
            self.summary = summary
            self.extended_summary = extended_summary
            
        return

    def __getitem__(self, key):
        if type(key) is not Perspective:
            raise TypeError('Key must be of type perspective')
        
        return self.perspectives[key]

    def __setitem__(self, key, value):
        if type(key) is not Perspective or type(value) is not Perspective:
            raise TypeError('Key and value must be of type perspective')
        
        self.perspectives[key] = value

    def __delitem__(self, key):
        if type(key) is not Perspective:
            raise TypeError('Key must be of type perspective')

        del self.perspectives[key]

    def __contains__(self, key):
        if type(key) is not Perspective:
            raise TypeError('Key must be of type perspective')
        
        return key in self.perspectives

    # Build a summary of the number of perspectives that need symbols
    # from a certain partition.
    def init_summary(self, p):
        num_perspectives = len(self.perspectives)

        # Build a simple summary that keeps track of how many symbols
        # that need a certain partition.
        self.summary = [num_perspectives for x in range(p.num_partitions)]

        # Build a more in-depth summary that keeps track of how many
        # perspectives need 1 symbol from a partition, how many need
        # 2, etc,
        self.extended_summary = [[0 for x in range(p.rows_per_partition + 1)] for y in range(p.num_partitions)]
        for i in range(p.num_partitions):
            self.extended_summary[i][-1] = num_perspectives

    def keys(self):
        return self.perspectives.keys()

    # Returns a shallow copy of itself
    def copy(self):
        perspectives = dict(self.perspectives)
        summary = list(self.summary)
        extended_summary = [list(x) for x in self.extended_summary]
        return BatchResult(perspectives, summary, extended_summary)
            
# Class representing a storage assignment. Contains efficient methods
# to evaluate changes in objective function etc.    
class Assignment(object):
    def __init__(self, p, X=None, A=None, score=None, index=None):
        self.p = p

        if X is None:
            self.X = np.zeros([p.num_batches, p.num_partitions])
        else:
            self.X = X

        if A is None:            
            self.A = [set() for x in range(p.num_servers)]
            self.label()
        else:
            self.A = A

        if score is None or index is None:
            self.build_index()
        else:
            self.score = score
            self.index = index

    def __str__(self):
        s = ''
        s = s + 'X:\n'
        s = s + str(self.X) + '\n'
        s = s + 'A:\n'
        s = s + str(self.A) + '\n'
        s = s + 'Score: ' + str(self.score)
        
        return s

    # Build an index pairing rows of the assignment matrix to which
    # perspectives they appear in. Only run once when creating a new
    # assignment.
    def build_index(self):
        self.score = 0
                
        # Index for which sets every row is contained in
        self.index = [BatchResult(self.p) for x in range(self.p.num_batches)]
        self.num_subsets = nchoosek(self.p.num_servers, self.p.q)        
        subsets = it.combinations(range(self.p.num_servers), self.p.q)

        # Build an index for which count vectors every row is part of
        for Q in subsets:
            for k in Q:
                rows = set()
                [rows.add(row) for row in self.A[k]]

                for j in range(self.p.sq, int(self.p.server_storage*self.p.q) + 1):
                    nu = set()
                    for subset in it.combinations([x for x in Q if x != k], j):
                        rows = rows | set.intersection(*[self.A[x] for x in subset])

                selector_vector = np.zeros([1, self.p.num_batches])
                for row in rows:
                    selector_vector[0][row] = 1

                count_vector = np.dot(selector_vector, self.X) - self.p.num_source_rows / self.p.num_partitions
                score = remainingUnicasts(count_vector)
                self.score = self.score + score

                perspective = Perspective(score, count_vector, rows)
                for row in rows:
                    assert perspective not in self.index[row]
                    self.index[row][perspective] = perspective

        # Initialize the summaries
        [x.init_summary(self.p) for x in self.index]

    # Label batches
    def label(self):
        assert self.p.server_storage * self.p.q % 1 == 0, 'Must be integer'
        labels =  it.combinations(range(self.p.num_servers), int(self.p.server_storage * self.p.q))

        # Randomize the labeling
        #labels = list(labels)
        #labels = random.sample(labels, len(labels))
        
        row = 0
        for label in labels:
            [self.A[x].add(row) for x in label]
            row = row + 1
        return

    # Compute a bound for this assignment
    def bound(self):
        b = 0
        for row_index in range(self.X.shape[0]):
            row = self.X[row_index]
            remaining_assignments = self.p.rows_per_batch - sum(row)
            assert remaining_assignments >= 0
            b = b + max(self.index[row_index].summary) * remaining_assignments

        # Bound can't be less than 0
        return max(self.score - b, 0)

    # Compute the bound based on the extended index. This bound is
    # tighter than the regular bound.
    def extended_bound(self):
        b = 0
        for row_index in range(self.X.shape[0]):
            row = self.X[row_index]
            remaining_assignments = self.p.rows_per_batch - sum(row)
            assert remaining_assignments >= 0

            # Make a copy of the extended index
            extended_summary = [list(x) for x in self.index[row_index].extended_summary]

            # Make assignments
            for i in range(int(remaining_assignments)):
                print(self.index[row_index].summary)
                print(extended_summary)
                print([sum(x[1:self.p.rows_per_partition + 1]) for x in extended_summary])
                print(max([sum(x[1:self.p.rows_per_partition + 1]) for x in extended_summary]))
                print('b', b)

                # Sum up the scores for all possible assignments
                score_sums = [sum(x[1:self.p.rows_per_partition + 1]) for x in extended_summary]

                # Select the best one
                best_assignment_score = max(score_sums)
                best_assignment_index = score_sums.index(best_assignment_score)
                
                b = b + best_assignment_score
                
                print('b', b)
                print('score', self.score)

                # Update the summary
                #for col_index in range(self.p.num_partitions):
                for count_index in range(self.p.rows_per_partition):
                    extended_summary[best_assignment_index][count_index] = extended_summary[best_assignment_index][count_index + 1]

                extended_summary[best_assignment_index][-1] = 0
                
                print(extended_summary)
                print()
        return max(self.score - b, 0)
                

    # Increment the element at [row, col] and update the objective
    # value. Returns a new assignment object. Does not change the
    # current assignment object.
    def increment(self, row, col):

        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        X = np.array(self.X)
        X[row, col] = X[row, col] + 1
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
            if new_perspective.count[0][col] == 0:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] = index[perspective_row].summary[col] - 1                       

            '''
            # Update the extended summary
            for perspective_row in new_perspective.rows:
                extended_summary = index[perspective_row].extended_summary

                # Find the first non-zero row
                for count_index in range(self.p.rows_per_partition, 0, -1):
                    if extended_summary[col][count_index] > 0:
                        break

                # Decrement the index by 1
                extended_summary[col][count_index] = extended_summary[col][count_index] - 1

                # Unless this is the last element, increment the
                # next by 1
                if count_index >= 1:
                    extended_summary[col][count_index - 1] = extended_summary[col][count_index - 1] + 1
            '''        
        
        # Return a new assignment object
        return Assignment(self.p, X=X, A=self.A, score=objective_value, index=index)

    # Decrement the element at [row, col] and update the objective
    # value. Returns a new assignment object. Does not change the
    # current assignment object.
    def decrement(self, row, col):
        assert self.X[row, col] >= 1, 'Can\'t decrement a value less than 1.'
        
        # Make a copy of the index
        index = [x.copy() for x in self.index]

        # Copy the assignment matrix and the objective value
        X = np.array(self.X)
        X[row, col] = X[row, col] - 1
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
            if new_perspective.count[0][col] == -1:
                for perspective_row in new_perspective.rows:
                    index[perspective_row].summary[col] = index[perspective_row].summary[col] + 1
        
        # Return a new assignment object
        return Assignment(self.p, X=X, A=self.A, score=objective_value, index=index)

# Branch-and-bound node
class Node(object):
    def __init__(self, p, assignment, row, symbols_separated):
        assert type(assignment) is Assignment
        assert type(row) is int
        assert type(symbols_separated) is list

        self.p = p
        
        self.assignment = assignment

        # Current row being assigned
        self.row = row

        # Remaining assignments for this row
        #self.remaining = remaining

        self.symbols_separated = symbols_separated

        self.id = uuid.uuid4()
        return

    def __hash__(self):
        return hash(self.id)

    # Return the number of remaining assignments for this row
    def remaining(self):
        return self.p.rows_per_batch - self.assignment.X[self.row].sum()
        
def branchAndBound(p, score=None, root=None, verbose=False):

    # Initialize the solver with a greedy solution
    if score is None:
        bestAssignment = assignmentGreedy(p)
        score = bestAssignment.score
    else:
        bestAssignment = None
        score = score

    if verbose:
        print('Starting B&B with score:', score)


    # Set of remaining nodes to explore
    # remainingNodes = set()
    remainingNodes = list()

    # Separate symbols by partition
    symbols_per_partition = p.num_coded_rows / p.num_partitions 
    symbols_separated = [symbols_per_partition for x in range(p.num_partitions)]

    # Add the root of the tree
    if root is None:
        root = Node(p, Assignment(p), 0, symbols_separated)
        
    # remainingNodes.add(root)
    remainingNodes.append(root)

    searched = 0
    pruned = 0

    while len(remainingNodes) > 0:
        node = remainingNodes.pop()
        searched = searched + 1

        # While there are no more valid assignments for this row, move
        # to the next one.
        while node.remaining() == 0:
            node.row = node.row + 1
            if node.row == p.num_batches:
                break
            
        # If there are no more assignments for this node, we've found
        # a solution.
        if node.row == p.num_batches:            
            #print('Found a solution:', node.assignment)
            #print('---------------------')
            
            # If this solution is better than the previous best, store it
            if node.assignment.score < score:
                bestAssignment = node.assignment
                score = bestAssignment.score

            continue

        # Iterate over all possible branches from this node
        for col in range(p.num_partitions):

            # Continue if there are no more valid assignments for this
            # column/partition            
            if node.symbols_separated[col] == 0:
                continue

            updatedAssignment = node.assignment.increment(node.row, col)

            # Continue if the bound for this assignment is no better
            # than the previous best solution.
            if updatedAssignment.bound() >= score:
                pruned = pruned + 1
                continue

            row = node.row

            # Decrement the count of remaining symbols for the
            # assigned partition.
            updatedSymbolsSeparated = list(node.symbols_separated)
            updatedSymbolsSeparated[col] = updatedSymbolsSeparated[col] - 1

            if verbose:
                print('Bound:', updatedAssignment.bound(),
                      'Row:', row,
                      '#nodes:', len(remainingNodes),
                      "Best:", score,
                      '#searched:', searched, '#pruned:', pruned)
            
            # Add the new node to the set of nodes to be explored
            # remainingNodes.add(Node(updatedAssignment, row, remaining, updatedSymbolsSeparated))
            remainingNodes.append(Node(p, updatedAssignment, row, updatedSymbolsSeparated))
            
    return bestAssignment            

def assignmentGreedy(p):
    assignment = Assignment(p)

    # Separate symbols by partition
    symbols_per_partition = p.num_coded_rows / p.num_partitions    
    symbols_separated = [symbols_per_partition for x in range(p.num_partitions)]

    # Assign symbols row by row
    for row in range(p.num_batches):
        #print('Assigning row', row, 'Bound:', assignment.bound())

        # Assign rows_per_batch rows per batch
        for i in range(p.rows_per_batch):
            score_best = 0
            best_col = 0

            # Try one column at a time
            for col in range(p.num_partitions):

                # If there are no more symbols left for this column to
                # assign, continue
                if symbols_separated[col] == 0:
                    continue

                # Evaluate the score of the assignment
                score_updated = assignment.index[row].summary[col] + symbols_separated[col] / symbols_per_partition

                # If it's better, store it
                if score_updated > score_best:
                    score_best = score_updated
                    best_col = col

            # Store the best assignment from this iteration
            assignment = assignment.increment(row, best_col)
            symbols_separated[best_col] = symbols_separated[best_col] - 1

    return assignment

def assignmentHybrid(p, clear=5, cutoff=1):

    # Start off with a greedy assignment
    print('Finding a candidate solution using the greedy solver.')
    assignment = assignmentGreedy(p)
    bestAssignment = assignment

    # Then iteratively try to improve it by de-assigning a random set
    # of batches and re-assigning them using the optimal
    # branch-and-bound solver.
    for c in range(2, clear + 1):
        print('Improving it by de-assigning', c, 'random rows at a time.')
        
        improvement = 100/c #  To make sure it runs at least this many times
        iterations = 1
        while improvement / iterations >= cutoff:
            score = bestAssignment.score
            assignment = bestAssignment
        
            # Keep track of the number of remaining symbols per partition    
            symbols_separated = [0 for x in range(p.num_partitions)]
    
            # Clear a few rows
            for i in range(c):
                row = random.randint(0, p.num_batches - 1)
                for col in range(p.num_partitions):
                    while assignment.X[row, col] > 0:                
                        assignment = assignment.decrement(row, col)
                        symbols_separated[col] = symbols_separated[col] + 1

            # Apply the branch-and-bound solver
            root = Node(p, assignment, 0, symbols_separated)
            newAssignment = branchAndBound(p, score=score, root=root)

            # If it found a better solution, overwrite the current one
            if newAssignment is not None:
                bestAssignment = newAssignment
                improvement = improvement + score - bestAssignment.score
                print('Iteration finished with an improvement of', score - bestAssignment.score)
                
            iterations = iterations + 1
                
    return bestAssignment

# Assign symbols randomly
def assignmentRandom(p, verbose=False):

    # Create a new assignment
    assignment = Assignment(p)

    # Then pick out X and A for to manipulate manually. We don't need
    # the indexing here.
    X = assignment.X
    A = assignment.A

    choices = random.sample(range(p.num_coded_rows), p.num_coded_rows)
    index = 0
    for row in range(p.num_batches):
        for col in range(p.rows_per_batch):
            X[row, choices[index] % p.num_partitions] = X[row, choices[index] % p.num_partitions] + 1
            index = index + 1
        
    return X, A

## Various performance measures
# Count the number of remaining unicasts
def remainingUnicasts(count_vector):
    unicasts = 0
    for e in count_vector[0]:
        if e < 0:
            unicasts = unicasts - e

    return unicasts

# Count the number of lost useful symbols in multicast rounds
def lostMulticasts(count_vector):
    lost_multicasts = 0
    for e in count_vector[0]:
        if e > 0:
            lost_multicasts = lost_multicasts + e

    return lost_multicasts

# Take the inner product with itself
def innerProduct(count_vector):
    return np.dot(count_vector, count_vector.T).sum()

# Length of the sum vector
def norm(count_vector):
    return math.sqrt(np.dot(count_vector, count_vector.T).sum())

# Sum of the absolute value of the vector
def absoluteSum(count_vector):
    return abs(count_vector).sum()

# The objective function to minimize
def objectiveFunction(X, assignment, p, Q=None, f=remainingUnicasts):
    
    # Count the total and worst-case score
    total_score = 0
    worst_score = 0

    # If a specific Q was given evaluate only that one.  Otherwise
    # evaluate all possible Q.
    if Q is None:
        subsets = it.combinations(range(p.num_servers), p.q)
        num_subsets = nchoosek(p.num_servers, p.q)
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
            [rows.add(row) for row in assignment[k]]

            for j in range(p.sq, int(p.server_storage*p.q) + 1):
                nu = set()
                for subset in it.combinations([x for x in Q if x != k], j):
                    rows = rows | set.intersection(*[assignment[x] for x in subset])

            selector_vector = np.zeros([1, p.num_batches])
            for row in rows:
                selector_vector[0][row] = 1

            count_vector = np.dot(selector_vector, X) - p.num_source_rows / p.num_partitions
            score = f(count_vector)
            
            total_score = total_score + score
            set_score = set_score + score

            #print('Q:', Q, 'Selected:', selector_vector, 'Score:', score)
            
        if set_score > worst_score:
            worst_score = set_score
            
    average_score = total_score / num_subsets
    #print('Average:', average_score, 'Worst:', worst_score)
    return average_score, worst_score

