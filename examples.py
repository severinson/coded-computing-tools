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

""" This is module contains coded used to run the simulations and generate the
plots we present in our paper. """

import math
import logging
import pandas as pd

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import model
from simulation import Simulator, SimulatorResult
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
import complexity
import rateless
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from solvers.hybrid import HybridSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

def load_data(parameters, labels, result):
    '''Load data from disk and construct a Pandas panel.

    Args:

    parameters: An array-like of parameters objects.

    labels: An array-like of same length as parameters with labels of
    the results loaded for each parameters object.

    result: A dict with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    Returns: A Pandas panel with the loaded date.

    '''
    assert len(parameters) == len(labels)
    assert isinstance(result, dict)
    directory = result['directory']
    data = dict()
    for par, label in zip(parameters, labels):
        filename = directory + par.identifier() + '.csv'

        # Try to load the data from disk
        try:
            dataframe = pd.read_csv(filename)
        except FileNotFoundError:
            logging.debug('No data for %s', filename)
            continue

        logging.debug('Loading data from %s', filename)

        # Scale map delay by its complexity
        if result['name'] == 'Uncoded':
            uncoded_storage = 1 / par.num_servers
            dataframe['delay'] *= complexity.matrix_vector_complexity(uncoded_storage * par.num_source_rows,
                                                                      par.num_columns)
        else:
            dataframe['delay'] *= complexity.matrix_vector_complexity(par.server_storage * par.num_source_rows,
                                                                      par.num_columns)

        # Normalize by number of source rows
        dataframe['delay'] /= par.num_source_rows

        # Compute load if not already in the DataFrame
        if 'load' not in dataframe:

            # Set load to inf if there are no results
            if ('unicast_load_1' not in dataframe or 'unicast_load_2' not in dataframe or
                'multicast_load_1' not in dataframe or 'multicast_load_2' not in dataframe):
                dataframe['load'] = math.inf

            # Otherwise select the strategy that minimizes the load
            elif (dataframe['unicast_load_1'].mean() + dataframe['multicast_load_1'].mean() <
                  dataframe['unicast_load_2'].mean() + dataframe['multicast_load_2'].mean()):
                dataframe['load'] = dataframe['unicast_load_1'] + dataframe['multicast_load_1']
            else:
                dataframe['load'] = dataframe['unicast_load_2'] + dataframe['multicast_load_2']

            # We are interested in load per source row
            # dataframe['load'] /= par.num_source_rows

        # Add the partitioned reduce delay
        dataframe['partitioned_reduce_delay'] = par.reduce_delay()

        # Add the uncoded reduce delay
        dataframe['uncoded_reduce_delay'] = 0

        # Add the unpartitioned reduce delay
        dataframe['rs_reduce_delay'] = par.reduce_delay(num_partitions=1)

        data[label] = dataframe

    # Create panel and return
    return pd.Panel.from_dict(data, orient='minor')

def lt_main():
    parameters = get_parameters_size_2()[:-2]
    plt.subplot('111')

    # delays = rateless.optimize_parameters(parameters, max_overhead=1.01)
    # df = pd.DataFrame(delays)
    # df['num_source_rows'] = [p.num_source_rows for p in parameters]
    # plt.loglog(df['num_source_rows'], df['lt'], 'g-', label='lt, 1.01')
    # plt.loglog(df['num_source_rows'], df['bdc'], 'g--', label='bdc, 1.01')

    # delays = rateless.optimize_parameters(parameters, max_overhead=1.02)
    # df = pd.DataFrame(delays)
    # df['num_source_rows'] = [p.num_source_rows for p in parameters]
    # plt.loglog(df['num_source_rows'], df['lt'], 'c-', label='lt, 1.02')
    # plt.loglog(df['num_source_rows'], df['bdc'], 'c--', label='bdc, 1.02')

    # delays = rateless.optimize_parameters(parameters, max_overhead=1.03)
    # df = pd.DataFrame(delays)
    # df['num_source_rows'] = [p.num_source_rows for p in parameters]
    # plt.loglog(df['num_source_rows'], df['lt'], 'm-', label='lt, 1.03')
    # plt.loglog(df['num_source_rows'], df['bdc'], 'm--', label='bdc, 1.03')

    delays = rateless.optimize_parameters(parameters, max_overhead=1.04)
    df = pd.DataFrame(delays)
    df['num_source_rows'] = [p.num_source_rows for p in parameters]
    plt.loglog(df['num_source_rows'], df['lt'], 'm-', label='lt, 1.04')
    plt.loglog(df['num_source_rows'], df['bdc'], 'm--', label='bdc, 1.04')

    delays = rateless.optimize_parameters(parameters, max_overhead=1.05)
    df = pd.DataFrame(delays)
    df['num_source_rows'] = [p.num_source_rows for p in parameters]
    plt.loglog(df['num_source_rows'], df['lt'], 'y-', label='lt, 1.05')
    plt.loglog(df['num_source_rows'], df['bdc'], 'y--', label='bdc, 1.05')

    delays = rateless.optimize_parameters(parameters, max_overhead=1.1)
    df = pd.DataFrame(delays)
    df['num_source_rows'] = [p.num_source_rows for p in parameters]
    plt.loglog(df['num_source_rows'], df['lt'], 'r-', label='lt, 1.1')
    plt.loglog(df['num_source_rows'], df['bdc'], 'r--', label='bdc, 1.1')

    delays = rateless.optimize_parameters(parameters, max_overhead=1.2)
    df = pd.DataFrame(delays)
    df['num_source_rows'] = [p.num_source_rows for p in parameters]
    plt.loglog(df['num_source_rows'], df['lt'], 'b-', label='lt, 1.2')
    plt.loglog(df['num_source_rows'], df['bdc'], 'b--', label='bdc, 1.2')

    delays = rateless.optimize_parameters(parameters, max_overhead=1.3)
    df = pd.DataFrame(delays)
    df['num_source_rows'] = [p.num_source_rows for p in parameters]
    plt.loglog(df['num_source_rows'], df['lt'], 'k-', label='lt, 1.3')
    plt.loglog(df['num_source_rows'], df['bdc'], 'k--', label='bdc, 1.3')

    plt.grid()
    plt.legend()
    plt.show()
    return

def main():
    '''Main examples function.'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning()
    size_parameters = get_parameters_size_2()

    # Setup the simulators
    heuristic_sim = Simulator(solver=HeuristicSolver(),
                              assignment_eval=sample_1000,
                              directory='./results/Heuristic/')

    random_sim = Simulator(solver=RandomSolver(), assignments=100,
                           assignment_eval=sample_100,
                           directory='./results/Random_100/')

    random_sim_1000 = Simulator(solver=RandomSolver(), assignments=1000,
                                assignment_eval=sample_1000,
                                directory='./results/Random_1000/')

    # hybrid_solver = HybridSolver(initialsolver=HeuristicSolver(),
    #                              directory='./saved_assignments_2/',
    #                              clear=3)
    # hybrid_solver = AssignmentLoader(directory='./results/Hybrid/assignments/')

    # hybrid_sim = Simulator(solver=hybrid_solver, assignments=1,
    #                        assignment_eval=sample_1000,
    #                        assignment_type=CachedAssignment,
    #                        directory='./results/Hybrid/')

    rs_sim = Simulator(solver=None, assignments=1,
                       parameter_eval=analytic.mds_performance,
                       directory='./results/RS/')

    uncoded_sim = Simulator(solver=None, assignments=1,
                            parameter_eval=analytic.uncoded_performance,
                            directory='./results/Uncoded/')

    # nostraggler_sim = Simulator(solver=None, assignments=1,
    #                             parameter_eval=analytic.nostraggler_performance,
    #                             directory='./results/Nostraggler/')

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    hybrid_partitions = hybrid_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    lt_partitions = lt_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)
    # nostraggler_partitions = nostraggler_sim.simulate_parameter_list(partition_parameters)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    hybrid_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    # lt_partitions.set_reduce_delay(complexity.lt_reduce_delay)
    lt_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_uncoded(enable=True)
    # nostraggler_partitions.set_reduce_delay(function=lambda x: 0)
    # nostraggler_partitions.set_nostraggler(enable=True)

    # Simulate size parameters
    # heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    # random_size = random_sim.simulate_parameter_list(size_parameters)
    # rs_size = rs_sim.simulate_parameter_list(size_parameters)
    # lt_size = lt_sim.simulate_parameter_list(size_parameters)
    # uncoded_size = uncoded_sim.simulate_parameter_list(size_parameters)
    # # nostraggler_size = nostraggler_sim.simulate_parameter_list(size_parameters)

    # Include the reduce delay
    # heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    # random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    # rs_size.set_reduce_delay(function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    # # lt_size.set_reduce_delay(complexity.lt_reduce_delay)
    # lt_partitions.set_reduce_delay(function=lambda x: 0)
    # uncoded_size.set_reduce_delay(function=lambda x: 0)
    # uncoded_size.set_uncoded(enable=True)
    # # nostraggler_size.set_reduce_delay(function=lambda x: 0)
    # # nostraggler_size.set_nostraggler(enable=True)

    # Setup plot settings
    heuristic_plot_settings = {
        'label': 'Heuristic',
        'color': 'r',
        'marker': 'H',
        'linewidth': 2,
        'size': 10}
    random_plot_settings = {
        'label': 'Random',
        'color': 'b',
        'marker': '^',
        'linewidth': 2,
        'size': 8}
    rs_plot_settings = {
        'label': 'RS',
        'color': 'c',
        'marker': 'v',
        'linewidth': 2,
        'size': 7}

    hybrid_plot_settings = {
        'label': 'Hybrid',
        'color': 'g',
        'marker': 's',
        'linewidth': 2,
        'size': 6}
    lt_plot_settings = {
        'label': 'LT',
        'color': 'g',
        'marker': 'v',
        'linewidth': 2,
        'size': 6}

    plot_settings = [heuristic_plot_settings, random_plot_settings,
                     lt_plot_settings, hybrid_plot_settings]

    # Plot including LT codes
    # load_delay_plot([heuristic_partitions, hybrid_partitions, random_partitions, lt_partitions],
    #                 [heuristic_plot_settings, hybrid_plot_settings, random_plot_settings, lt_plot_settings],
    #                 'partitions', xlabel='T', normalize=rs_partitions)


    # load_delay_plot([heuristic_size, random_size, lt_size],
    #                 [heuristic_plot_settings, random_plot_settings, lt_plot_settings],
    #                 'servers', xlabel='Servers $K$', normalize=rs_size)

    # Plots without LT codes.
    # load_delay_plot([heuristic_partitions, hybrid_partitions, random_partitions],
    #                 [heuristic_plot_settings, hybrid_plot_settings, random_plot_settings],
    #                 'partitions', xlabel='$T$', normalize=rs_partitions)

    # load_delay_plot([heuristic_size, random_size],
    #                 [heuristic_plot_settings, random_plot_settings],
    #                 'servers', xlabel='$K$', normalize=rs_size)

    # Plots against the uncoded performance
    # load_delay_plot([heuristic_partitions, hybrid_partitions, random_partitions, rs_partitions],
    #                 [heuristic_plot_settings, hybrid_plot_settings, random_plot_settings, rs_plot_settings],
    #                 'partitions', xlabel='$T$', normalize=uncoded_partitions)

    # load_delay_plot([heuristic_size, random_size, rs_size],
    #                 [heuristic_plot_settings, random_plot_settings, rs_plot_settings],
    #                 'servers', xlabel='$K$', normalize=uncoded_size)


    # Plots for presentation
    rs_plot_settings = {
        'label': 'RS (Li \emph{et al.})',
        'color': 'r',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}
    nostraggler_plot_settings = {
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
    load_delay_plot([rs_partitions, heuristic_partitions],
                    [rs_plot_settings, heuristic_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)

    # load_delay_plot([rs_size, heuristic_size, random_size],
    #                 [rs_plot_settings, heuristic_plot_settings, random_plot_settings],
    #                 'servers', xlabel='Servers $K$', normalize=uncoded_size)
    return

    # Presentation settings
    rs_plot_settings = {
        'label': 'RS (Li \emph{et al.})',
        'color': 'r',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}

    heuristic_plot_settings = {
        'label': 'BDC (Us)',
        'color': 'b',
        'marker': '-s',
        'linewidth': 2,
        'size': 7}

    load_delay_plot([nostraggler_size],
                    [nostraggler_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)

    load_delay_plot([rs_size, nostraggler_size],
                    [rs_plot_settings, nostraggler_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)

    load_delay_plot([rs_size, nostraggler_size, heuristic_size],
                    [rs_plot_settings, nostraggler_plot_settings, heuristic_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)

    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
