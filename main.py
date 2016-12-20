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
        s = s + 'Multicast messages: ' + str(self.multicast_load() * self.num_source_rows * self.num_outputs) + '\n'
        s = s + 'Unpartitioned load: ' + str(self.unpartitioned_load()) + '\n'
        s = s + '------------------------------'        
        return s


    # Calculate S_q as defined in Li2016
    def s_q(self):
        muq = int(self.server_storage * self.q)        
        s = 0
        for j in range(muq, 0, -1):
            s = s + self.B_j(j)

            if s > (1 - self.server_storage):
                return max(min(j + 1, muq), 1)
            elif s == (muq):
                return max(j, 1)

        return 1

    # Calculate B_j as defined in Li2016
    def B_j(self, j):
        B_j = nchoosek(self.q-1, j) * nchoosek(self.num_servers-self.q, int(self.server_storage*self.q)-j)
        B_j = B_j / (self.q / self.num_servers) / nchoosek(self.num_servers, int(self.server_storage * self.q))
        return B_j

    # The communication load after only multicasting
    def multicast_load(self):
        load = 0
        for j in range(self.sq, int(self.server_storage*self.q)):
            load = load + self.B_j(j+1) / (j+1)

        return load
        
    # Calculate the total communication load per output vector  for an
    # unpartitioned storage design as defined in Li2016.
    def unpartitioned_load(self, L2=False):
        muq = int(self.server_storage * self.q)

        # Load assuming remaining values are unicast
        L_1 = 1 - self.server_storage
        for j in range(self.sq, muq + 1):
            Bj = self.B_j(j)
            L_1 = L_1 + Bj / j
            L_1 = L_1 - Bj

        # Load assuming more multicasting is done
        if self.sq > 1:
            L_2 = 0
            for j in range(self.sq - 1, muq + 1):
                L_2 = L_2 + self.B_j(j) / j
        else:
            L_2 = math.inf        

        # Return the minimum if L2 is enabled. Otherwise return L_1        
        if L2:
            return min(L_1, L_2)
        else:
            return L_1

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

# Evaluate the performance of the greedy solver.
# Performance is evaluated by counting the total number of remaining
# needed values.
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

    average_load = total_load / n
    report = ''
    report = report + 'Simulation Results:\n'
    report = report + '------------------------------\n'    
    report = report + 'Num runs: ' + str(n) + '\n'
    report = report + 'Max: ' + str(max_load) + '\n'
    report = report + 'Avg.: ' + str(average_load) + '\n'
    report = report + '------------------------------'
    print(report)
