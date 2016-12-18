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

# This is the main file for all code relating to the paper by Albin
# Severinson, Alexandre Graell i Amat, and Eirik Rosnes. If you want
# to re-create the simulations in the paper, or want to simulate the
# system for other parameters, this is where you start.

# General modules
import numpy as np
import scipy as sp
import pandas as pd
from scipy.misc import comb as nchoosek
import itertools as it
import copy
import random
import math
import uuid

# Package-specific modules
import simulator
import solver

# System parameters representation. This struct is used as an argument
# for most functions, and creating one should be the first step in
# simulating the system for some parameters.
class SystemParameters(object):
    def __init__(self, rows_per_batch, num_servers, q, num_outputs, server_storage, num_partitions):
        # TODO: Raise exceptions instead of assertions?
        assert type(rows_per_batch) is int
        assert type(num_servers) is int
        assert type(q) is int
        assert type(num_outputs) is int
        assert type(server_storage) is float or server_storage == 1
        assert type(num_partitions) is int
        
        self.num_servers = num_servers
        self.q = q
        self.num_outputs = num_outputs
        self.server_storage = server_storage
        self.num_partitions = num_partitions
        self.rows_per_batch = rows_per_batch

        assert num_outputs / q % 1 == 0, 'num_outputs must be divisible by the number of servers.'
        assert server_storage * q % 1 == 0, 'server_storage * q must be integer.'
        
        num_batches = nchoosek(num_servers, server_storage*q)
        assert num_batches % 1 == 0, 'There must be an integer number of batches.'
        num_batches = int(num_batches)
        self.num_batches = num_batches

        num_coded_rows = rows_per_batch * num_batches
        self.num_coded_rows = num_coded_rows

        num_source_rows = q / num_servers * num_coded_rows
        assert num_source_rows % 1 == 0, 'There must be an integer number of source rows.'
        num_source_rows = int(num_source_rows)
        self.num_source_rows = num_source_rows

        assert num_source_rows / num_partitions % 1 == 0, 'There must be an integer number of source rows per partition.'
        rows_per_partition = int(num_source_rows / num_partitions)
        self.rows_per_partition = rows_per_partition

        self.sq = self.s_q()

    def __str__(self):
        s = '------------------------------\n'
        s = s + 'System Parameters:\n'
        s = s + '------------------------------\n'
        s = s + 'Rows per batch:  ' + str(self.rows_per_batch) + '\n'
        s = s + 'Servers: ' + str(self.num_servers) + '\n'
        s = s + 'Wait for: ' + str(self.q) + '\n'
        s = s + 'Number of outputs: ' + str(self.num_outputs) + '\n'
        s = s + 'Server storage: ' + str(self.server_storage) + '\n'
        s = s + 'Number of batches: ' + str(self.num_batches) + '\n'
        s = s + 'Batches: ' + str(self.num_batches) + '\n'
        s = s + '|T|: ' + str(self.server_storage * self.q) + '\n'
        s = s + 'Code rate: ' + str(self.q / self.num_servers) + '\n'
        s = s + 'Source rows: ' + str(self.num_source_rows) + '\n'
        s = s + 'Coded rows: ' + str(self.num_coded_rows) + '\n'
        s = s + 'Partitions: ' + str(self.num_partitions) + '\n'
        s = s + 'Rows per partition: ' + str(self.rows_per_partition) + '\n'
        s = s + 's_q: ' + str(self.sq) + '\n'
        s = s + '------------------------------'        
        return s

    # Calculate the symbols received per multicast round. Defined in Li2016.    
    def symbols_received_per_round(self, multicast_index):
        code_rate = self.q / self.num_servers
        num_batches = nchoosek(self.num_servers, int(self.server_storage*self.q))
        return nchoosek(self.q - 1, multicast_index) * nchoosek(self.num_servers-self.q, int(self.server_storage*self.q)-multicast_index) * self.rows_per_batch;

    # Calculate the minimum multicast index for which we need to
    # multicast, s_q. Defined in Li2016.
    def s_q(self):
        needed_symbols = (1 - self.server_storage) * self.num_source_rows
        for multicast_index in range(int(self.server_storage * self.q), 0, -1):
            needed_symbols = needed_symbols - self.symbols_received_per_round(multicast_index)
            if needed_symbols <= 0:
                return max(multicast_index - 1, 1)
        return 1

    # Calculate B_j as defined in Li2016
    # TODO: Make sure this works as intended    
    def B_j(self, j):
        B_j = nchoosek(self.q-1, j) * nchoosek(self.num_servers-self.q, int(self.server_storage*self.q)-j)
        B_j = B_j / (self.q / self.num_servers) / nchoosek(self.num_servers, int(self.server_storage * self.q))
        return B_j

    def multicast_load(self):
        load = 0
        for j in range(self.sq, int(self.server_storage*self.q)):
            print(j+1)
            load = load + self.B_j(j+1) / (j+1)

        return load
        
    # Calculate the total communication load per output vector  for an
    # unpartitioned storage design as defined in Li2016.
    # TODO: Make sure this works as intended    
    def unpartitioned_load(q, num_servers, server_storage, num_source_rows):
        code_rate = self.q / self.num_servers
        num_batches = nchoosek(self.num_servers, int(self.server_storage*self.q))
        rows_per_batch = self.num_source_rows / self.code_rate / self.num_batches    
        #sq = self.s_q(q, num_servers, server_storage, num_source_rows)

        L_1 = 1 - server_storage
        for j in range(self.sq, int(self.server_storage*self.q) + 1):
            Bj = self.B_j(j)
            L_1 = L_1 + Bj / j
            L_1 = L_1 - Bj

        L_2 = 0
        for j in range(self.sq-1, int(self.server_storage*self.q) + 1):
            L_2 = L_2 + B_j(j)
        
        print('L_1:', L_1, 'L_2:', L_2, 'Load per output vector.')
        return min(L_1, L_2)    

# Main function with some usage examples
def main():

    '''
    p = SystemParameters(2, # Rows per batch
                         6, # Number of servers (K)
                         4, # Servers to wait for (q)
                         4, # Outputs (N)
                         1/2, # Server storage (\mu)
                         1) # Partitions (T)
    print(p)
    print(p.multicast_load())
    return
    #assignment = solver.assignmentHybrid(p)
    #print(solver.objectiveFunction(assignment.X, assignment.A, p))
    #print(assignment)
    #T = [1, 2, 5, 10]    
    #randomizedPerformance(p, T)
    assignment = solver.Assignment(p)
    X = np.array([[2, 0, 0, 0, 0],
                   [2, 0, 0, 0, 0],
                   [2, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 0, 2, 0, 0],
                   [0, 0, 2, 0, 0],
                   [0, 0, 2, 0, 0],
                   [0, 0, 0, 2, 0],
                   [0, 0, 0, 2, 0],
                   [0, 0, 0, 2, 0],
                   [0, 0, 0, 0, 2],
                   [0, 0, 0, 0, 2],
                   [0, 0, 0, 0, 2]])
    print(solver.objectiveFunction(X, assignment.A, p, Q=(0,1,2,3)))
    return
    '''

    # Example usage in evaluating the performance of a scheme using
    # our proposed solvers.    

    p = SystemParameters(56, # Rows per batch
                         9, # Number of servers (K)
                         2, # Servers to wait for (q)
                         2, # Outputs (N)
                         1/2, # Server storage (\mu)
                         1) # Partitions (T)
    T = [1, 2, 4, 7, 8, 14, 16, 28, 56]
    results = hybridPerformance(p, T)

    #greedyPerformance(p, T)
    #randomizedPerformance(p, T)
    return results


    
    p = SystemParameters(2, # Rows per batch
                         6, # Number of servers (K)
                         4, # Servers to wait for (q)
                         4, # Outputs (N)
                         1/2, # Server storage (\mu)
                         5) # Partitions (T)
    assignment = solver.assignmentGreedy(p)
    print(assignment)
    return
    
    T = [1, 2, 5, 10]
    greedyPerformance(p, T)
    randomizedPerformance(p, T)

    # Example usage of the branch-and-bound solver. The solver is only
    # useable for very small problem instances.
    '''
    bar = branchAndBound(p)
    print(bar)    
    print(objectiveFunction(bar.X, bar.A, p))    
    return
    '''

    # Evaluate the performance of an assignment by simulating the
    # shuffling scheme.
    assignment = assignmentGreedy(p)
    simulate(p, assignment.X, assignment.A)

    return

# Evaluate the performance of the greedy assignment solver.
# Performance is evaluated by counting the total number of remaining
# needed values.
def greedyPerformance(p, partitions, n=100):
    print('Calculating average greedy performance for', partitions, 'partitions.')
    print(p)

    results = list()
    for T in partitions:

        # Create a system
        p_T =  SystemParameters(p.rows_per_batch, p.num_servers, p.q, p.num_outputs, p.server_storage, T)
        
        totalAverage = 0
        totalWorst = 0    
        for i in range(n):
            assignment = solver.assignmentGreedy(p_T)
            score = solver.objectiveFunction(assignment.X, assignment.A, p_T)
            totalAverage = totalAverage + score[0]
            totalWorst = totalWorst + score[1]

        print('Partitions:', T, 'Average:', totalAverage / n, 'Worst:', totalWorst / n)
        results.append((totalAverage/n, totalWorst/n))

    print('Partitions:', partitions)
    print('Results:', results)        
    return    

# Evaluate the performance of a random assignment.
# Performance is evaluated by counting the total number of remaining
# needed values.
def randomizedPerformance(p, partitions, n=100):
    print('Calculating average randomized performance for', partitions, 'partitions.')
    print(p)

    load = dict()
    for T in partitions:

        # Create a system
        p_T =  SystemParameters(p.rows_per_batch, p.num_servers, p.q, p.num_outputs, p.server_storage, T)
        
        totalAverage = 0
        totalWorst = 0    
        for i in range(n):
            X, A = solver.assignmentRandom(p_T)
            score = solver.objectiveFunction(X, A, p_T)
            totalAverage = totalAverage + score[0]
            totalWorst = totalWorst + score[1]

        print('Partitions:', T, 'Average:', totalAverage / n, 'Worst:', totalWorst / n)
        load[T] = dict()
        load[T]['average'] = totalAverage/n
        load[T]['worst'] = totalWorst/n

    results = pd.DataFrame(load)
    print(results)
    return results

def hybridPerformance(p, partitions, n=1):
    print('Calculating hybrid performance for', partitions, 'partitions.')
    print(p)

    load = dict()
    for T in partitions:

        # Create a system
        p_T =  SystemParameters(p.rows_per_batch, p.num_servers, p.q, p.num_outputs, p.server_storage, T)
        
        totalAverage = 0
        totalWorst = 0    
        for i in range(n):
            assignment = solver.assignmentHybrid(p_T, clear=3)
            print(assignment)
            score = solver.objectiveFunction(assignment.X, assignment.A, p_T)
            totalAverage = totalAverage + score[0]
            totalWorst = totalWorst + score[1]

        print('Partitions:', T, 'Average:', totalAverage / n, 'Worst:', totalWorst / n)
        load[T] = dict()
        load[T]['average'] = totalAverage/n
        load[T]['worst'] = totalWorst/n

    results = pd.DataFrame(load)
    print(results)
    return results

# Simulate the performance of a given assignment
def simulate(p, X, A, n=100):

    total_load = 0
    max_load = 0
    results = list()
    for i in range(n):

        # Create a computation system model
        system = simulator.ComputationSystem(p)

        # Assign symbols according to some assignment
        system.storage_from_assignment(X, A)

        # Perform the map phase
        selected = system.map()

        # Perform the shuffle phase
        # TODO: Do we need psi?
        load, psi = system.shuffle()

        # Update stats
        total_load = total_load + load
        max_load = max(max_load, load)
        results.append(load)

    average_load = total_load / num_runs
    report = ''
    report = report + 'Simulation Results:\n'
    report = report + '------------------------------\n'    
    report = report + 'Num runs: ' + str(num_runs) + '\n'
    report = report + 'Max: ' + str(max_load) + '\n'
    report = report + 'Avg.: ' + str(average_load) + '\n'
    report = report + '------------------------------'
    print(report)
