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
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import model
import simulation
from evaluation import analytic
from evaluation import independent
import mathtools
import complexity
from solvers import randomsolver
from solvers import heuristicsolver
from solvers import treesolver

def load_delay_plot(parameters, results, xlabel=''):
    '''Create a plot with two subplots for load and delay respectively.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    xlabel: X axis label

    '''
    assert isinstance(parameters, list)
    assert isinstance(results, list)
    assert isinstance(xlabel, str)

    _ = plt.figure(figsize=(8,5))

    # Plot load
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold', visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    communication_load_plot(parameters, results, ylabel='Load', subplot=True)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=3, mode="expand", borderaxespad=0., numpoints=1)

    plt.legend(numpoints=1, fontsize=18, loc='best')

    # Plot delay
    ax2 = plt.subplot(212, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters, results, yindex='delay',
                             xlabel=xlabel, ylabel='Delay', subplot=True,
                             include_reduce=True)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.show()
    return

def delay_delay_plot(parameters1, parameters2, results):
    '''Create a figure with two subplots for the delay of two parameter
    lists.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    xlabel: X axis label

    '''
    assert isinstance(parameters1, list)
    assert isinstance(parameters2, list)
    assert isinstance(results, list)

    _ = plt.figure()

    # Plot delay for parameters1
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters1, results, yindex='delay',
                             xlabel='$K$', ylabel='Delay', subplot=True,
                             include_reduce=True)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., numpoints=1)

    # Plot delay for parameters2
    ax2 = plt.subplot(212)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters2, results, yindex='delay',
                             xlabel='$T$', ylabel='Delay',
                             subplot=True, include_reduce=True)
    # plt.legend(numpoints=1, fontsize=20, loc='best')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)
    plt.show()
    return

def load_load_plot(parameters1, parameters2, results):
    '''Create a figure with two subplots for the delay of two parameter
    lists.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    xlabel: X axis label

    '''
    assert isinstance(parameters1, list)
    assert isinstance(parameters2, list)
    assert isinstance(results, list)

    _ = plt.figure()

    # Plot delay for parameters1
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    communication_load_plot(parameters1, results, xlabel='$K$',
                            ylabel='Load', subplot=True)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., numpoints=1)

    # Plot delay for parameters2
    ax2 = plt.subplot(212)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    communication_load_plot(parameters2, results, xlabel='$T$',
                            ylabel='Load', subplot=True)
    # plt.legend(numpoints=1, fontsize=20, loc='best')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)
    plt.show()
    return

def computational_delay_plot(parameters, results, yindex=None,
                             xlabel='', ylabel='',
                             include_reduce=False, normalize=True, subplot=False):
    '''Create a computational delay plot from an array-like of parameters
    and results.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    yindex: The index of the Y axis data. For example 'batches', or 'delay'.

    xlabel: X axis label.

    ylabel: Y axis label.

    include_reduce: Include the reduce time if True. yindex must be
    set to 'delay' if this is True.

    subplot: Set to True if the plot is intended to be a subplot. This
    will keep it from creating a new plot window, creating a legend,
    and automatically showing the plot.

    '''
    assert isinstance(yindex, str)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    if include_reduce:
        assert yindex == 'delay', 'Reduce delay can only be included with the delay yindex.'

    if not subplot:
        _ = plt.figure()

    plt.grid(True, which='both')
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.autoscale()

    for result in results:
        labels = [par.num_partitions for par in parameters]
        panel = load_data(parameters, labels, result)
        if yindex not in panel:
            continue

        if include_reduce and result['reduce_label'] not in panel:
            continue

        xdata = panel.minor_axis
        ydata = panel[yindex]

        if include_reduce:
            ydata += panel[result['reduce_label']]

        if normalize:
            ydata /= panel['uncoded_delay']

        ydata_mean = ydata.mean()
        ydata_min = ydata.min() #quantile(0.05)
        ydata_max = ydata.max() #quantile(0.95)
        yerr = np.array([np.array(ydata_mean - ydata_min), np.array(ydata_max - ydata_mean)])

        plt.semilogx(xdata, ydata_mean, result['color'] + result['marker'], label=result['name'], linewidth=2)
        plt.errorbar(xdata, ydata_mean, yerr=yerr, fmt='none', ecolor=result['color'])

    if not subplot:
        plt.legend(numpoints=1, fontsize=20, loc='best')
        plt.show()

    return

def communication_load_plot(parameters, results, xlabel='', ylabel='',
                            subplot=False, normalize=True):
    '''Create a communication load plot from an array-like of parameters
    and results.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    xlabel: X axis label.

    ylabel: Y axis label.

    subplot: Set to True if the plot q is intended to be a subplot. This
    will keep it from creating a new plot window, creating a legend,
    and automatically showing the plot.

    '''
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    if not subplot:
        _ = plt.figure()

    plt.grid(True, which='both')
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.autoscale()

    for result in results:
        labels = [par.num_partitions for par in parameters]
        panel = load_data(parameters, labels, result)
        if 'load' not in panel:
            continue

        xdata = panel.minor_axis
        ydata = panel['load']

        if normalize:
            ydata /= panel['uncoded_load']

        ydata_mean = ydata.mean()
        ydata_min = ydata.min()
        ydata_max = ydata.max()
        yerr = np.array([ydata_mean - ydata_min, ydata_max - ydata_mean])

        plt.semilogx(xdata, ydata_mean, result['color'] + result['marker'],
                     label=result['name'], linewidth=2)
        plt.errorbar(xdata, ydata_mean, yerr=yerr, fmt='none', ecolor=result['color'])

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    if not subplot:
        plt.legend(numpoints=1, fontsize=20, loc='best')
        plt.show()

    return

def load_data(parameters, labels, result, shuffling_strategy='best'):
    '''Load data from disk and construct a Pandas panel.

    Args:
    parameters: An array-like of parameters objects.

    labels: An array-like of same length as parameters with labels of
    the results loaded for each parameters object.

    result: A dict with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    shuffling_strategy: Can be '1', '2', or best.

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

        # Compute uncoded performance for normalization
        dataframe['uncoded_load'] = pd.DataFrame([par.num_source_rows * (1 - par.server_storage)])
        dataframe['uncoded_load'] *= par.num_outputs / par.q
        dataframe['uncoded_load'] /= par.num_source_rows
        dataframe['uncoded_delay'] = pd.DataFrame([par.computational_delay(q=par.num_servers)])
        dataframe['uncoded_delay'] *= complexity.matrix_vector_complexity(par.server_storage * par.num_source_rows,
                                                                          par.num_columns)
        dataframe['uncoded_delay'] /= par.num_source_rows# * par.num_outputs

        # Scale map delay by its complexity
        dataframe['delay'] *= complexity.matrix_vector_complexity(par.server_storage * par.num_source_rows,
                                                                  par.num_columns)
        # Normalize by number of source rows and output vectors
        dataframe['delay'] /= par.num_source_rows# * par.num_outputs

        # Set load
        # dataframe['load'] = dataframe['unicasts_strat_1']
        # dataframe['load'] += dataframe['multicasts_strat_1']
        # dataframe['load'] /= par.num_source_rows * par.num_outputs

        # dataframe['load'] = dataframe['unicasts_strat_2']
        # dataframe['load'] += dataframe['multicasts_strat_2']
        # dataframe['load'] /= par.num_source_rows * par.num_outputs

        # Compute load if not already stored
        if 'load' not in dataframe:

            # Set load to inf if there are no results
            if ('unicasts_strat_1' not in dataframe or 'unicasts_strat_2' not in dataframe or
                'multicasts_strat_1' not in dataframe or 'multicasts_strat_2' not in dataframe):
                dataframe['load'] = math.inf

            # Otherwise select the strategy that minimizes the load
            elif (dataframe['unicasts_strat_1'].mean() + dataframe['multicasts_strat_1'].mean() <
                  dataframe['unicasts_strat_2'].mean() + dataframe['multicasts_strat_2'].mean()):
                dataframe['load'] = dataframe['unicasts_strat_1'] + dataframe['multicasts_strat_1']

            else:
                dataframe['load'] = dataframe['unicasts_strat_2'] + dataframe['multicasts_strat_2']

            # We are interested in load per source row and output vector
            dataframe['load'] /= par.num_source_rows * par.num_outputs

        # Add the partitioned reduce delay
        dataframe['partitioned_reduce_delay'] = par.reduce_delay()

        # Add the uncoded reduce delay
        dataframe['uncoded_reduce_delay'] = 0

        # Add the unpartitioned reduce delay
        dataframe['rs_reduce_delay'] = par.reduce_delay(num_partitions=1)

        # Add the peeling decoder reduce delay
        dataframe['peeling_reduce_delay'] = par.reduce_delay_ldpc_peeling()

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
        # print(par.server_storage * par.num_source_rows * par.num_columns, par.num_outputs, par.num_servers / par.q)
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
    """Main examples function."""

    # Compute/simulate the load and delay for various parameters and solvers
    heuristic_simulator = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                               directory='./results/HeuristicSolver/',
                                               num_assignments=1, num_samples=1000,
                                               rerun=False)

    random_simulator = simulation.Simulator(solver=randomsolver.RandomSolver(),
                                            directory='./results/RandomSolver/',
                                            num_assignments=10, num_samples=100,
                                            rerun=False)

    # unsupervised_simulator = simulation.Simulator(solver=None,
    #                                               par_eval=evaluation.eval_unsupervised,
    #                                               directory='./results/Unsupervised/',
    #                                               num_samples=100)

    heuristic_upper_bound = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                                 rerun=True, directory='./results/HeuristicUpper/',
                                                 eval_fun=analytic.block_diagonal_upper_bound,
                                                 num_assignments=1)

    heuristic_analytic = simulation.Simulator(solver=None, rerun=False,
                                              par_eval=analytic.average_heuristic,
                                              directory='./results/HeuristicAnalytic/')

    tree_simulator = simulation.Simulator(solver=treesolver.TreeSolver(),
                                          directory='./results/Tree/', num_assignments=1,
                                          num_samples=1000)

    lt_code = simulation.Simulator(solver=None, rerun=False,
                                   par_eval=analytic.lt_performance,
                                   directory='./results/LT/')

    rs_code = simulation.Simulator(solver=None, rerun=False,
                                   par_eval=analytic.mds_performance,
                                   directory='./results/RS/')

    uncoded = simulation.Simulator(solver=None, rerun=True,
                                   par_eval=analytic.uncoded_performance,
                                   directory='./results/Uncoded/')

    # Get parameter lists
    load_delay_parameters = get_parameters_size()[0:-2]
    load_delay_parameters_light = load_delay_parameters#[0:-2]
    partitioning_parameters = get_parameters_partitioning()

    # Load-delay parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=load_delay_parameters_light)
    random_simulator.simulate_parameter_list(parameter_list=load_delay_parameters_light)
    # unsupervised_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    # heuristic_upper_bound.simulate_parameter_list(parameter_list=load_delay_parameters_light)
    # heuristic_analytic.simulate_parameter_list(parameter_list=load_delay_parameters)
    lt_code.simulate_parameter_list(parameter_list=load_delay_parameters)
    rs_code.simulate_parameter_list(parameter_list=load_delay_parameters)
    uncoded.simulate_parameter_list(parameter_list=load_delay_parameters)

    # Partitioning parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    random_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    # unsupervised_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    # heuristic_upper_bound.simulate_parameter_list(parameter_list=partitioning_parameters)
    # heuristic_analytic.simulate_parameter_list(parameter_list=partitioning_parameters)
    lt_code.simulate_parameter_list(parameter_list=partitioning_parameters)
    rs_code.simulate_parameter_list(parameter_list=partitioning_parameters)
    uncoded.simulate_parameter_list(parameter_list=partitioning_parameters)

    # tree_simulator.simulate(partitioning_parameters[29])
    # for parameters, i in zip(partitioning_parameters, range(len(partitioning_parameters))):
    #     print('Simulation {}:\n{}'.format(i, parameters))
    #     tree_simulator.simulate(parameters)

    # tree_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)

    # Setup a list of dicts containing plotting parameters
    plot_settings = list()
    plot_settings.append({'directory': './results/RS/',
                          'reduce_label': 'rs_reduce_delay',
                          'color': 'k', 'marker': '-', 'name': 'RS'}) # H

    # plot_settings.append({'directory': './results/Uncoded/',
    #                       'reduce_label': 'uncoded_reduce_delay',
    #                       'color': 'b', 'marker': '-', 'name': 'Uncoded'}) # s

    # plot_settings.append({'directory': './results/LT/',
    #                       'reduce_label': 'peeling_reduce_delay',
    #                       'color': 'm', 'marker': '--', 'name': 'LT'}) s

    # plot_settings.append({'directory': './results/HeuristicAnalytic/',
    #                       'reduce_label': 'partitioned_reduce_delay',
    #                       'color': 'b', 'marker': 'D', 'name': 'Heuristic Theo.'})

    plot_settings.append({'directory': './results/HeuristicSolver/',
                          'reduce_label': 'partitioned_reduce_delay',
                          'color': 'r', 'marker': 'd', 'name': 'BDC, Heuristic'})

    # plot_settings.append({'directory': './results/Unsupervised/',
    #                       'color': 'k', 'marker': 'D', 'name':
    #                       'Unsupervised'})

    # plot_settings.append({'directory': './results/HeuristicUpper/',
    #                       'reduce_label': 'partitioned_reduce-delay',
    #                       'color': 'm', 'marker': 'D', 'name':
    #                       'Heuristic Upper Bound'})

    # plot_settings.append({'directory': './results/Tree/',
    #                       'reduce_label': 'partitioned_reduce_delay',
    #                       'color': 'm', 'marker': 'D', 'name':
    #                       'Tree'})

    plot_settings.append({'directory': './results/RandomSolver/',
                          'reduce_label': 'partitioned_reduce_delay',
                          'color': 'c', 'marker': 'v', 'name': 'BDC, Random'})

    # hist_plot(partitioning_parameters[29], {'directory': './results/HeuristicSolverHist/',
    #                                         'reduce_label': 'partitioned_reduce_delay',
    #                                         'color': 'r', 'marker': 'd', 'name': 'Heuristic'})

    # hist_plot(partitioning_parameters[29], {'directory': './results/Tree/',
    #                                         'reduce_label': 'partitioned_reduce_delay',
    #                                         'color': 'm', 'marker': 'D', 'name':
    #                                         'Tree'})


    load_delay_plot(partitioning_parameters, plot_settings, xlabel='$T$')
    load_delay_plot(load_delay_parameters, plot_settings, xlabel='$K$')
    # delay_delay_plot(load_delay_parameters, partitioning_parameters, plot_settings)
    # delay_delay_plot(load_delay_parameters, partitioning_parameters, plot_settings)
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
