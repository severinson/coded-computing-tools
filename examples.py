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

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model
import simulation
import evaluation
from solvers import randomsolver
from solvers import heuristicsolver

def load_delay_plot(parameters, results):
    '''Create a plot with two subplots for load and delay respectively.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    '''
    _ = plt.figure()

    # Plot load
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold', visible=False)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    communication_load_plot(parameters, results, ylabel='Load', subplot=True)
    plt.legend(numpoints=1, fontsize=20, loc='best')

    # Plot delay
    ax2 = plt.subplot(212, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters, results, yindex='delay',
                             xlabel='$T$', ylabel='Delay', subplot=True)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()
    return

def delay_delay_plot(parameters1, parameters2, results):
    '''Create a figure with two subplots for the delay of two parameter
    lists.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    '''
    _ = plt.figure()

    # Plot delay for parameters1
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters1, results, yindex='delay',
                             xlabel='$K$', ylabel='Delay', subplot=True)

    # Plot delay for parameters2
    ax2 = plt.subplot(212)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    computational_delay_plot(parameters2, results, yindex='delay',
                             xlabel='$T$', ylabel='Delay',
                             subplot=True)
    plt.legend(numpoints=1, fontsize=20, loc='best')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)
    plt.show()
    return

def computational_delay_plot(parameters, results, yindex=None,
                             xlabel='', ylabel='',
                             include_reduce=True, subplot=False):
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
        xdata = panel.minor_axis
        ydata = panel[yindex]
        if include_reduce:
            ydata += panel[result['reduce_label']]

        ydata_mean = ydata.mean()
        ydata_min = ydata.quantile(q=0.05)
        ydata_max = ydata.quantile(0.95)
        yerr = np.array([ydata_mean - ydata_min, ydata_max - ydata_mean])

        plt.semilogx(xdata, ydata_mean, result['color'] + result['marker'], label=result['name'])
        plt.errorbar(xdata, ydata_mean, yerr=yerr, fmt='none', ecolor=result['color'])

    if not subplot:
        plt.legend(numpoints=1, fontsize=20, loc='best')
        plt.show()

    return

def communication_load_plot(parameters, results, xlabel='', ylabel='',
                            subplot=False):
    '''Create a communication load plot from an array-like of parameters
    and results.

    Args:
    parameters: An array-like of parameters objects.

    result: An array-like of dicts with plot parameter. Example:
    {'directory': './results/RandomSolver/', 'color': 'c', 'marker': 'v', 'name': 'Random'}

    xlabel: X axis label.

    ylabel: Y axis label.

    subplot: Set to True if the plot is intended to be a subplot. This
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
        xdata = panel.minor_axis
        ydata_mean = panel['load'].mean()
        ydata_min = panel['load'].quantile(q=0.05)
        ydata_max = panel['load'].quantile(0.95)
        yerr = np.array([ydata_mean - ydata_min, ydata_max - ydata_mean])

        plt.semilogx(xdata, ydata_mean, result['color'] + result['marker'], label=result['name'])
        plt.errorbar(xdata, ydata_mean, yerr=yerr, fmt='none', ecolor=result['color'])

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

        # We need to add the number of multicasts manually
        dataframe['load'] += par.num_multicasts()

        # We are interested in load per source row and output vector
        dataframe['load'] /= par.num_source_rows * par.num_outputs

        # Add the partitioned reduce delay
        dataframe['partitioned_reduce_delay'] = par.reduce_delay()

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
    directory = result['directory']
    filename = directory + par.identifier() + '.csv'

    try:
        dataframe = pd.read_csv(filename)
    except FileNotFoundError:
        logging.warning('No data for %s!', filename)
        return

    _ = plt.figure()
    ax = plt.axes()
    plt.hist(dataframe['batches'], bins=50, normed=True)
    plt.grid(True, which='both')
    plt.autoscale()
    plt.ylabel('Probability Density', fontsize=20)
    plt.xlabel('Batches Required', fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=20, weight='bold')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.show()
    return

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

def main():
    """Main examples function."""

    # Compute/simulate the load and delay for various parameters and solvers
    heuristic_simulator = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                               directory='./results/HeuristicSolverHist/',
                                               num_assignments=1000, num_samples=1,
                                               verbose=True)

    random_simulator = simulation.Simulator(solver=randomsolver.RandomSolver(),
                                            directory='./results/RandomSolver/',
                                            num_assignments=100, num_samples=100,
                                            verbose=True)

    unsupervised_simulator = simulation.Simulator(solver=None,
                                                  par_eval=evaluation.eval_unsupervised,
                                                  directory='./results/Unsupervised/',
                                                  num_samples=100)

    # heuristic_upper_bound = simulation.Simulator(solver=None, rerun=False,
    #                                              par_eval=evaluation.upper_bound_heuristic,
    #                                              directory='./results/HeuristicUpper/')

    heuristic_avg = simulation.Simulator(solver=None, rerun=True,
                                         par_eval=evaluation.average_heuristic,
                                         directory='./results/HeuristicAverage/')

    lt_code = simulation.Simulator(solver=None, rerun=False,
                                   par_eval=evaluation.lt_performance,
                                   directory='./results/LT/')

    rs_code = simulation.Simulator(solver=None, rerun=False,
                                   par_eval=evaluation.mds_performance,
                                   directory='./results/RS/')

    # Get parameter lists
    load_delay_parameters = get_parameters_load_delay()
    partitioning_parameters = get_parameters_partitioning()

    # Load-delay parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    # random_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    unsupervised_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    # heuristic_upper_bound.simulate_parameter_list(parameter_list=load_delay_parameters)
    heuristic_avg.simulate_parameter_list(parameter_list=load_delay_parameters)
    lt_code.simulate_parameter_list(parameter_list=load_delay_parameters)
    rs_code.simulate_parameter_list(parameter_list=load_delay_parameters)

    # Partitioning parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    random_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    unsupervised_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    # heuristic_upper_bound.simulate_parameter_list(parameter_list=partitioning_parameters)
    heuristic_avg.simulate_parameter_list(parameter_list=partitioning_parameters)
    lt_code.simulate_parameter_list(parameter_list=partitioning_parameters)
    rs_code.simulate_parameter_list(parameter_list=partitioning_parameters)

    # Setup a list of dicts containing plotting parameters
    plot_settings = list()
    # plot_settings.append({'directory': './results/HybridSolver/',
    #                       'color': 'k', 'marker': 's', 'name':
    #                       'Hybrid'})

    plot_settings.append({'directory': './results/RandomSolver/',
                          'reduce_label': 'partitioned_reduce_delay',
                          'color': 'c', 'marker': 'v', 'name': 'Random'})

    plot_settings.append({'directory': './results/HeuristicSolverHist/',
                          'reduce_label': 'partitioned_reduce_delay',
                          'color': 'r', 'marker': 'd', 'name': 'Heuristic'})

    # plot_settings.append({'directory': './results/Unsupervised/',
    #                       'color': 'k', 'marker': 'D', 'name':
    #                       'Unsupervised'})

    # plot_settings.append({'directory': './results/HeuristicUpper/',
    #                       'color': 'm', 'marker': 'H', 'name': 'Heuristic Upper Bound'})

    plot_settings.append({'directory': './results/HeuristicAverage/',
                          'reduce_label': 'partitioned_reduce_delay',
                          'color': 'b', 'marker': 'o', 'name': 'Heuristic Theo.'})

    plot_settings.append({'directory': './results/LT/',
                          'reduce_label': 'peeling_reduce_delay',
                          'color': 'g', 'marker': '>', 'name': 'LT Peeling'})

    plot_settings.append({'directory': './results/RS/',
                          'reduce_label': 'rs_reduce_delay',
                          'color': 'k', 'marker': 'D', 'name': 'Reed-Solomon'})

    load_delay_plot(partitioning_parameters, plot_settings)
    delay_delay_plot(load_delay_parameters, partitioning_parameters, plot_settings)
    return

if __name__ == '__main__':
    main()
