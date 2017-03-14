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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model
import simulation
import evaluation
from solvers import randomsolver
from solvers import heuristicsolver

def load_delay_plots(parameters, results,
                     xdata_name='partitions', ydata_name='servers', xlabel=None,
                     normalize=False, include_reduce=False):
    '''Plot load and delay.

    Attempts to load results from disk and will skip any missing results.

    Args:
    parameters: The parameters for which to plot.

    results: A dict containing info regarding the results to plot. The
    dict must contain entries 'directory', 'color', 'marker', and
    'name'.

    xdata_name: What data to plot on the x axis. Can be 'partitions'
    or 'servers'.

    ydata_name: What data to plot on the y axis of the delay plot. Can
    be 'batches' 'servers', or 'delay'.

    xlabel: The label of the x axis.

    normalize: The load and delay are normalized with the performance
    of the unpartitioned (MDS code) performance if this argument is
    True.

    include_reduce: The reduce phase latency is included if this
    argument is True.

    '''

    assert isinstance(xdata_name, str)
    assert isinstance(ydata_name, str)

    # Setup a dict for storing the data to plot
    data = dict()
    for par in parameters:
        for result in results:
            # for directory, key in zip(directories, keys):
            directory = result['directory']
            name = result['name']

            data[name] = dict()
            data[name]['load'] = dict()
            data[name]['load']['mean'] = list()
            data[name]['load']['min'] = list()
            data[name]['load']['max'] = list()

            data[name]['delay'] = dict()
            data[name]['delay']['mean'] = list()
            data[name]['delay']['min'] = list()
            data[name]['delay']['max'] = list()

            data[name]['partitions'] = list()
            data[name]['servers'] = list()

    # Load the data from disk
    for par in parameters:
        for result in results:
            directory = result['directory']
            name = result['name']

            try:
                df = pd.read_csv(directory + par.identifier() + '.csv')
            except FileNotFoundError:
                print('No data for', directory + '/' + par.identifier())
                continue

            if ydata_name not in df:
                print('No data with label', ydata_name, 'for', directory + '/' + par.identifier())
                continue

            load = df['load'] + par.num_multicasts()
            load /= par.num_source_rows * par.num_outputs
            if normalize:
                load /= par.unpartitioned_load()

            data[name]['load']['mean'].append(load.mean())
            data[name]['load']['min'].append(load.min())
            data[name]['load']['max'].append(load.max())

            delay = df[ydata_name]
            if include_reduce:
                delay += par.reduce_delay()

            if normalize and include_reduce:
                delay /= par.computational_delay() + par.reduce_delay(num_partitions=1)
            elif normalize and not include_reduce:
                delay /= par.computational_delay()

            data[name]['delay']['mean'].append(delay.mean())
            data[name]['delay']['min'].append(delay.min())
            data[name]['delay']['max'].append(delay.max())

            data[name]['partitions'].append(par.num_partitions)
            data[name]['servers'].append(par.num_servers)

    # Plot the load
    # ----------------------------------------------
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.grid(True, which='both')
    plt.ylabel('Comm. Load Increase', fontsize=18)
    plt.autoscale()

    for result in results:
        name = result['name']
        color = result['color']
        marker = result['marker']

        xdata = np.array(data[name][xdata_name])
        load = data[name]['load']
        load_mean = np.array(load['mean'])
        load_min = np.array(load['min'])
        load_max = np.array(load['max'])
        plt.semilogx(xdata, load_mean, color + marker, label=name)
        yerr = np.array([load_mean - load_min, load_max - load_mean])
        plt.errorbar(xdata, load_mean, yerr=yerr, fmt='none', ecolor=color)

    plt.legend(numpoints=1, fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Plot the delay
    # ----------------------------------------------
    ax2 = plt.subplot(212, sharex=ax1)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', weight='bold')
    #plt.rc('font', weight='bold')
    plt.grid(True, which='both')
    plt.ylabel('Comp. Delay Increase', fontsize=18)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=20)

    plt.autoscale()

    for result in results:
        name = result['name']
        color = result['color']
        marker = result['marker']

        xdata = np.array(data[name][xdata_name])
        delay = data[name]['delay']
        delay_mean = np.array(delay['mean'])
        delay_min = np.array(delay['min'])
        delay_max = np.array(delay['max'])
        plt.semilogx(xdata, delay_mean, color + marker, label=name)
        yerr = np.array([delay_mean - delay_min, delay_max - delay_mean])
        plt.errorbar(xdata, delay_mean, yerr=yerr, fmt='none', ecolor=color)

    #plt.legend()
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), fontsize=20, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=20)

    # Misc plot settings
    # ----------------------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()

    return

def get_parameters_load_delay():
    """ Get a list of parameters for the load-delay plot. """

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
    """ Get a list of parameters for the partitioning plot. """

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
    """ Main examples function """

    ## Run simulations for various solvers and parameters
    heuristic_simulator = simulation.Simulator(solver=heuristicsolver.HeuristicSolver(),
                                               directory='./results/HeuristicSolver/',
                                               verbose=True)

    random_simulator = simulation.Simulator(solver=randomsolver.RandomSolver(),
                                            directory='./results/RandomSolver/',
                                            num_assignments=100, verbose=True)

    unsupervised_simulator = simulation.Simulator(solver=None,
                                                  par_eval=evaluation.eval_unsupervised,
                                                  directory='./results/Unsupervised/',
                                                  num_samples=100)

    heuristic_upper_bound = simulation.Simulator(solver=None, rerun=False,
                                                 par_eval=evaluation.upper_bound_heuristic,
                                                 directory='./results/HeuristicUpper/')

    heuristic_avg = simulation.Simulator(solver=None, rerun=True,
                                         par_eval=evaluation.average_heuristic,
                                         directory='./results/HeuristicAverage/')

    # Get parameter lists
    load_delay_parameters = get_parameters_load_delay()
    partitioning_parameters = get_parameters_partitioning()

    # Load-delay parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    random_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    unsupervised_simulator.simulate_parameter_list(parameter_list=load_delay_parameters)
    # heuristic_upper_bound.simulate_parameter_list(parameter_list=load_delay_parameters)

    for par in partitioning_parameters:
        heuristic_upper_bound.simulate(par)

    for par in load_delay_parameters:
        heuristic_avg.simulate(par)

    # Partitioning parameters
    heuristic_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    random_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    unsupervised_simulator.simulate_parameter_list(parameter_list=partitioning_parameters)
    # heuristic_bound.simulate_parameter_list(parameter_list=partitioning_parameters)n
    for par in partitioning_parameters:
        heuristic_upper_bound.simulate(par)

    for par in partitioning_parameters:
        heuristic_avg.simulate(par)

    # Setup a list of dicts containing plotting parameters
    plot_settings = list()
    # plot_settings.append({'directory': './results/HybridSolver/',
    #                       'color': 'k', 'marker': 's', 'name': 'Hybrid'})
    plot_settings.append({'directory': './results/RandomSolver/',
                          'color': 'b', 'marker': 'o', 'name': 'Random'})
    plot_settings.append({'directory': './results/HeuristicSolver/',
                          'color': 'r', 'marker': 'd', 'name': 'Heuristic'})
    plot_settings.append({'directory': './results/Unsupervised/',
                          'color': 'k', 'marker': 'x', 'name': 'Unsupervised'})
    plot_settings.append({'directory': './results/HeuristicUpper/',
                          'color': 'm', 'marker': 'H', 'name': 'Heuristic Upper Bound'})
    plot_settings.append({'directory': './results/HeuristicAverage/',
                          'color': 'b', 'marker': 'o', 'name': 'Heuristic Average'})

    load_delay_plots(load_delay_parameters, plot_settings, xdata_name='servers', xlabel='$K$')
    load_delay_plots(partitioning_parameters, plot_settings, xdata_name='partitions', xlabel='$T$')
    return

if __name__ == '__main__':
    main()
