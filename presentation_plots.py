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

'''This script is used to generate the plots for the thesis defense.

'''

import logging
from matplotlib import rc
import matplotlib.pyplot as plt
import model
from simulation import Simulator
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
import complexity
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from examples import load_delay_plot

# Set default plot fonts
rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)

def get_parameters_size():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = int(1e4)
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(rows_per_server=rows_per_server,
                                                                 rows_per_partition=rows_per_partition,
                                                                 min_num_servers=servers,
                                                                 code_rate=code_rate,
                                                                 muq=muq, num_columns=num_columns)
        parameters.append(par)

    return parameters

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
        par = model.SystemParameters(rows_per_batch=rows_per_batch, num_servers=num_servers, q=q,
                                     num_outputs=num_outputs, server_storage=server_storage,
                                     num_partitions=partitions)
        parameters.append(par)

    return parameters

def main():
    '''Main examples function.'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning()
    size_parameters = get_parameters_size()[0:-2]

    # Setup the simulators
    heuristic_sim = Simulator(solver=HeuristicSolver(),
                              assignment_eval=sample_1000,
                              directory='./results/Heuristic/')

    random_sim = Simulator(solver=RandomSolver(), assignments=100,
                           assignment_eval=sample_100,
                           directory='./results/Random_100/')

    rs_sim = Simulator(solver=None, assignments=1,
                       parameter_eval=analytic.mds_performance,
                       directory='./results/RS/')

    uncoded_sim = Simulator(solver=None, assignments=1,
                            parameter_eval=analytic.uncoded_performance,
                            directory='./results/Uncoded/')

    cmapred_sim = Simulator(solver=None, assignments=1,
                            parameter_eval=analytic.cmapred_performance,
                            directory='./results/Cmapred/')

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)
    cmapred_partitions = cmapred_sim.simulate_parameter_list(partition_parameters)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(function=lambda x:
                                   complexity.partitioned_reduce_delay(x, partitions=1))
    uncoded_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_uncoded(enable=True)
    cmapred_partitions.set_reduce_delay(function=lambda x: 0)
    cmapred_partitions.set_cmapred(enable=True)

    # Simulate size parameters
    heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    random_size = random_sim.simulate_parameter_list(size_parameters)
    rs_size = rs_sim.simulate_parameter_list(size_parameters)
    uncoded_size = uncoded_sim.simulate_parameter_list(size_parameters)
    cmapred_size = cmapred_sim.simulate_parameter_list(size_parameters)

    # Include the reduce delay
    heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    uncoded_size.set_reduce_delay(function=lambda x: 0)
    uncoded_size.set_uncoded(enable=True)
    cmapred_size.set_reduce_delay(function=lambda x: 0)
    cmapred_size.set_cmapred(enable=True)

    # Plots for presentation
    rs_plot_settings = {
        'label': r'RS (Li \emph{et al.})',
        'color': 'r',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}
    cmapred_plot_settings = {
        'label': 'Coded MapReduce',
        'color': 'g',
        'marker': 'v',
        'linewidth': 2,
        'size': 7}
    heuristic_plot_settings = {
        'label': 'BDC, Heuristic (Us)',
        'color': 'b',
        'marker': '-s',
        'linewidth': 2,
        'size': 7}
    random_plot_settings = {
        'label': 'BDC, Random (Us)',
        'color': 'g',
        'marker': '-^',
        'linewidth': 2,
        'size': 8}

    # The unified scheme
    load_delay_plot([rs_size],
                    [rs_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/unified_1.pdf')

    load_delay_plot([rs_size, cmapred_size],
                    [rs_plot_settings, cmapred_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/unified_2.pdf')

    rs_size.set_reduce_delay(function=lambda x:
                             complexity.partitioned_reduce_delay(x, partitions=1))

    load_delay_plot([rs_size, cmapred_size],
                    [rs_plot_settings, cmapred_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/unified_3.pdf')

    # Numerical results, size
    load_delay_plot([rs_size],
                    [rs_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/size_1.pdf')

    load_delay_plot([rs_size, random_size],
                    [rs_plot_settings, random_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/size_2.pdf')

    load_delay_plot([rs_size, random_size, heuristic_size],
                    [rs_plot_settings, random_plot_settings, heuristic_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/presentation/size_3.pdf')

    # Numerical results, partitions
    load_delay_plot([rs_partitions],
                    [rs_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/presentation/partitions_1.pdf')

    load_delay_plot([rs_partitions, random_partitions],
                    [rs_plot_settings, random_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/presentation/partitions_2.pdf')

    load_delay_plot([rs_partitions, random_partitions, heuristic_partitions],
                    [rs_plot_settings, random_plot_settings, heuristic_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/presentation/partitions_3.pdf')
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
