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

'''This module is an application used for performance profiling.

'''

import os
import tempfile
import model
import simulation
from solvers.heuristicsolver import HeuristicSolver
from solvers.hybrid import HybridSolver
from evaluation import sampled
from evaluation.binsearch import SampleEvaluator

def get_parameters_partitioning():
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

def get_parameters_load_delay():
    '''Get a list of parameters for the load-delay plot.'''

    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(rows_per_server,
                                                                 rows_per_partition,
                                                                 servers, code_rate,
                                                                 muq)
        parameters.append(par)

    return parameters

def time_multicast_size():
    for parameters in get_parameters_load_delay():
        for _ in range(10000):
            try:
                _ = parameters.multicast_set_size_1()
            except model.ModelError:
                pass

            try:
                _ = parameters.multicast_set_size_2()
            except model.ModelError:
                pass

def time_hybrid():
    parameters = get_parameters_partitioning()[10]
    solver = HybridSolver(initialsolver=HeuristicSolver())
    assignment = solver.solve(parameters)
    return

@profile
def time_binsearch():
    parameters = get_parameters_load_delay()[1]
    solver = HeuristicSolver()
    assignment = solver.solve(parameters)
    evaluator = SampleEvaluator(num_samples=1000)
    result = evaluator.evaluate(parameters, assignment)
    return

def main():
    parameters = get_parameters_load_delay()
    with tempfile.TemporaryDirectory() as tmpdirname:
        heuristic_simulator = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                                   directory=tmpdirname,
                                                   num_assignments=1, num_samples=1,
                                                   rerun=True)
        heuristic_simulator.simulate(parameters[-1])
    return

if __name__ == '__main__':
    time_binsearch()
