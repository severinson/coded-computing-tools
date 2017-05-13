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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import model
from simulation import Simulator, SimulatorResult
from evaluation import analytic
from evaluation import independent
from evaluation.binsearch import SampleEvaluator
import mathtools
import complexity
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from solvers.treesolver import TreeSolver
from solvers.hybrid import HybridSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

def load_delay_plot(results, plot_settings, xlabel='', normalize=None):
    '''Create a plot with two subplots for load and delay respectively.

    Args:

    results: SimulatorResult to plot.

    plot_settings: List of dicts with plot settings.

    xlabel: X axis label

    normalize: If a SimulatorResult is provided, all ploted results
    are normalized by this one.

    '''
    assert isinstance(results, list)
    assert isinstance(plot_settings, list)
    assert isinstance(normalize, SimulatorResult) or normalize is None

    _ = plt.figure(figsize=(8,5))

    # Plot load
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold', visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    for result, plot_setting in zip(results, plot_settings):
        plot_result(result, plot_setting, 'partitions', 'load',
                    ylabel='Load', subplot=True, normalize=normalize)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.legend(numpoints=1, fontsize=18, loc='best')

    # Plot delay
    ax2 = plt.subplot(212, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    for result, plot_setting in zip(results, plot_settings):
        plot_result(result, plot_setting, 'partitions', 'delay', xlabel=xlabel,
                    ylabel='Delay', subplot=True, normalize=normalize)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()
    return

def plot_result(result, plot_settings, xdata, ydata, xlabel='',
                ylabel='', subplot=False, normalize=None,
                errorbars=False):
    '''Plot simulated results.

    Args:

    result: A SimulationResult.

    plot_settings: A dict with plot settings.

    xdata: Label of the X axis data ('partitions' or 'servers').

    ydata: Label of the Y axis data ('load' or 'delay').

    xlabel: X axis label.

    ylabel: Y axis label.

    subplot: Set to True if the plot q is intended to be a subplot.
    This will keep it from creating a new plot window, creating a
    legend, and automatically showing the plot.

    normalize: Normalize the plotted data by that of these results.
    Must be a list of SimulationResults of length equal to results.

    errorbars: Plot error bars.

    '''
    assert isinstance(result, SimulatorResult)
    assert isinstance(plot_settings, dict)
    assert xdata == 'partitions' or xdata == 'servers'
    assert ydata == 'load' or ydata == 'delay'
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(subplot, bool)
    assert isinstance(normalize, SimulatorResult) or normalize is None

    if not subplot:
        _ = plt.figure()

    plt.grid(True, which='both')
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.autoscale()

    label = plot_settings['label']
    color = plot_settings['color']
    style = color + plot_settings['marker']
    linewidth = plot_settings['linewidth']
    size = plot_settings['size']

    xarray = result[xdata]
    ymean = result[ydata][0, :]
    ymin = result[ydata][1, :]
    ymax = result[ydata][2, :]
    yerr = np.zeros([2, len(ymean)])
    yerr[0, :] = ymean - ymin
    yerr[1, :] = ymax - ymean
    if normalize is not None:
        ymean /= normalize[ydata][0, :]
        yerr[0, :] /= normalize[ydata][0, :]
        yerr[1, :] /= normalize[ydata][0, :]

    plt.semilogx(xarray, ymean, style, label=label,
                 linewidth=linewidth, markersize=size)
    if errorbars:
        plt.errorbar(xarray, ymean, yerr=yerr, fmt='none', ecolor=color)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    if not subplot:
        plt.legend(numpoints=1, fontsize=20, loc='best')
        plt.show()

    return

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

def hist_plot(par, result):
    '''Draw a histogram for some par and result objects.

    Args:
    par: A parameters object.

    result: A dict with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    '''
    data = load_data([par], [par.num_partitions], result)
    _ = plt.figure()
    ax = plt.axes()
    print(data)
    plt.hist(data['batches'], bins=15, normed=True)
    plt.grid(True, which='both')
    plt.autoscale()
    plt.ylabel('Probability Density', fontsize=20)
    plt.xlabel('Batches', fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=20, weight='bold')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.show()
    return

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
        print(par)

    return parameters

def get_parameters_size_2():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 5
    code_rate = 8/10
    muq = 2
    num_columns = int(1e4)
    parameters = list()
    num_servers = [0]
    for servers in num_servers:
        print(servers)
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

def get_parameters_partitioning_2():
    '''Get a list of parameters for the partitioning plot.'''
    rows_per_batch = 250
    num_servers = 9
    q = 8
    num_outputs = q
    server_storage = 1/4
    num_partitions = [2, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000]

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(rows_per_batch=rows_per_batch, num_servers=num_servers, q=q,
                                     num_outputs=num_outputs, server_storage=server_storage,
                                     num_partitions=partitions)
        parameters.append(par)

    return parameters

def get_parameters_partitioning_3():
    '''Get a list of parameters for the partitioning plot.'''
    rows_per_batch = 500
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/6
    max_partitions = 1500
    num_partitions = list()
    for i in range(1, 1500 + 1):
        if max_partitions / i % 1 == 0:
            num_partitions.append(int(max_partitions / i))

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(rows_per_batch=rows_per_batch, num_servers=num_servers, q=q,
                                     num_outputs=num_outputs, server_storage=server_storage,
                                     num_partitions=partitions, num_columns=None)
        parameters.append(par)

    return parameters

def foo():
    df = pd.read_csv('./results2/Heuristic/m_6000_K_9_q_6_N_6_muq_2_T_3000.csv')
    df['load'] = df['unicast_load_1'] + df['multicast_load_1']
    print(df['load'].mean())
    print(df[df['load'] < 3.1]['delay'].mean(), df[df['load'] > 3.1]['delay'].mean())
    _ = plt.figure()

    plt.hist(df[df['load'] < 3.1]['delay'], normed=True, alpha=0.7)
    plt.hist(df[df['load'] > 3.1]['delay'], normed=True, alpha=0.7)
    plt.show()

def main():
    '''Main examples function.'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning()
    size_parameters = get_parameters_size()[0:-2]
    partition_parameters_3 = get_parameters_partitioning_3()

    # Setup the simulators
    heuristic_sim = Simulator(solver=HeuristicSolver(),
                              assignment_eval=sample_1000,
                              directory='./results2/Heuristic/')

    random_sim = Simulator(solver=RandomSolver(), assignments=10,
                           assignment_eval=sample_100,
                           directory='./results2/Random/')

    hybrid_solver = HybridSolver(initialsolver=HeuristicSolver(),
                                 directory='./saved_assignments_2/',
                                 clear=3)
    hybrid_solver = AssignmentLoader(directory='./results/HybridSolver')
    hybrid_sim = Simulator(solver=hybrid_solver, assignments=1,
                           assignment_eval=sample_1000,
                           assignment_type=CachedAssignment,
                           directory='./results2/Hybrid/')

    rs_sim = Simulator(solver=None, assignments=1,
                       parameter_eval=analytic.mds_performance,
                       directory='./results2/RS/')

    uncoded_sim = Simulator(solver=None, assignments=1,
                            parameter_eval=analytic.uncoded_performance,
                            directory='./results2/Uncoded/')

    # Run the simulations
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    hybrid_partitions = hybrid_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)

    # heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters_3)
    # random_partitions = random_sim.simulate_parameter_list(partition_parameters_3)
    # hybrid_partitions = hybrid_sim.simulate_parameter_list(partition_parameters_3)
    # rs_partitions = rs_sim.simulate_parameter_list(partition_parameters_3)
    # uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters_3)

    heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    random_size = random_sim.simulate_parameter_list(size_parameters)
    rs_size = rs_sim.simulate_parameter_list(size_parameters)
    uncoded_size = uncoded_sim.simulate_parameter_list(size_parameters)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    hybrid_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    uncoded_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_uncoded(enable=True)

    heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_size.set_reduce_delay(function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    uncoded_size.set_reduce_delay(function=lambda x: 0)
    uncoded_size.set_uncoded(enable=True)

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
        'size': 12}
    hybrid_plot_settings = {
        'label': 'Hybrid',
        'color': 'g',
        'marker': 's',
        'linewidth': 2,
        'size': 6}

    plot_settings = [heuristic_plot_settings, random_plot_settings,
                     hybrid_plot_settings]

    load_delay_plot([heuristic_partitions, random_partitions, hybrid_partitions],
                    plot_settings, xlabel='T', normalize=rs_partitions)

    load_delay_plot([heuristic_size, random_size],
                    plot_settings, xlabel='K', normalize=rs_size)
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
