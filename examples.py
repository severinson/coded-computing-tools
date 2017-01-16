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
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model
import evaluation
from solvers import randomsolver
from solvers import heuristicsolver

def simulate(parameters, solver, directory, num_runs, num_samples=100, verbose=False):
    """ Run simulations for the parameters returned by get_parameters()

    Writes the results to disk as .csv files. Will also write the best
    assignment matrix found to disk in the same directory.

    Args:
    parameters: List of SystemParameters for which to run simulations.
    solver: The solver to use. A function pointer.
    directory: The directory to store the results in.
    num_runs: The number of simulations to run.
    num_samples: The number of objective function samples.
    verbose: Passed to the solver.
    """
    directory += solver.identifier + '/'

    # Create the directory to store the results in if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for par, i in zip(parameters, range(1, len(parameters) + 1)):

        # Try to load the results from disk
        try:
            pd.read_csv(directory + par.identifier() + '.csv')
            print('Found results for', directory, par.identifier(),
                  'on disk. Skipping.',
                  '(' + str(i) + '/' + str(len(parameters)) + ')')
            continue

        # Run the simulations if we couldn't find any
        except FileNotFoundError:
            print('Running simulations for', directory, par.identifier(),
                  '(' + str(i) + '/' + str(len(parameters)) + ')')

            best_assignment = None
            best_avg_load = math.inf

            results = list()
            for _ in range(num_runs):

                # Create an assignment
                assignment = solver.solve(par, verbose=verbose)
                assert model.is_valid(par, assignment.assignment_matrix, verbose=True)

                # Evaluate it
                avg_load, avg_delay = evaluation.objective_function_sampled(par,
                                                                            assignment.assignment_matrix,
                                                                            assignment.labels,
                                                                            num_samples=num_samples)

                # Store the results
                result = dict()
                result['load'] = avg_load
                result['delay'] = avg_delay
                results.append(result)

                # Keep the best assignment
                if avg_load < best_avg_load:
                    best_assignment = assignment
                    best_avg_load = avg_load

                # Write a dot to show that progress is being made
                sys.stdout.write('.')
                sys.stdout.flush()

            print('')

            # Create a pandas dataframe and write it to disk
            df = pd.DataFrame(results)
            df.to_csv(directory + par.identifier() + '.csv')

            # Write the best assignment to disk
            assignment.save(directory=directory)
    return

def partitioning_plots(parameters):
    """ Plot load and delay against number of partitions.

    Attempts to load results from disk and will skip any missing results.

    Args:
    parameters: The parameters for which to plot.
    """

    # List of directories with results
    directories = ['./results/random/', './results/heuristic/']
    colors = ['b', 'r']
    markers = ['o', 'd']
    #keys = ['random', 'hybrid', 'block']
    keys = ['Random', 'Heuristic']

    # Setup somewhere to store the results we need
    results = dict()
    for par in parameters:
        for directory, key in zip(directories, keys):
            results[key] = dict()
            result = results[key]

            result['load'] = dict()
            result['load']['mean'] = list()
            result['load']['min'] = list()
            result['load']['max'] = list()

            result['delay'] = dict()
            result['delay']['mean'] = list()
            result['delay']['min'] = list()
            result['delay']['max'] = list()

            result['partitions'] = list()

    # Load the results from disk
    for par in parameters:
        for directory, key in zip(directories, keys):
            try:
                df = pd.read_csv(directory + par.identifier() + '.csv')
            except FileNotFoundError:
                print('No data for', directory, par.identifier())
                continue

            result = results[key]

            load = df['load'] + par.num_multicasts()
            load /= par.num_source_rows * par.num_outputs * par.unpartitioned_load()
            result['load']['mean'].append(load.mean())
            result['load']['min'].append(load.min())
            result['load']['max'].append(load.max())

            delay = df['delay'] / par.computational_delay()
            result['delay']['mean'].append(delay.mean())
            result['delay']['min'].append(delay.min())
            result['delay']['max'].append(delay.max())

            result['partitions'].append(par.num_partitions)

    # Plot the load
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.grid(True, which='both')
    plt.ylabel('Comm. Load Increase', fontsize=15)
    plt.autoscale()

    for key, color, marker in zip(keys, colors, markers):
        partitions = np.array(results[key]['partitions'])
        load = results[key]['load']
        load_mean = np.array(load['mean'])
        load_min = np.array(load['min'])
        load_max = np.array(load['max'])
        plt.semilogx(partitions, load_mean, color + marker, label=key)
        yerr = np.array([load_mean - load_min, load_max - load_mean])
        plt.errorbar(partitions, load_mean, yerr=yerr, fmt='none', ecolor=color)

    x1, x2, y1, y2 = plt.axis()
    plt.axis([x1, x2, 0.95, 1.2])
    plt.legend(loc='upper left', numpoints=1)
    plt.setp(ax1.get_yticklabels(), fontsize=10)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Plot the delay
    ax2 = plt.subplot(212, sharex=ax1)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', weight='bold')
    #plt.rc('font', weight='bold')
    plt.grid(True, which='both')
    plt.ylabel('Comp. Delay Increase', fontsize=15)
    plt.xlabel('Partitions', fontsize=15)
    plt.autoscale()

    for key, color, marker in zip(keys, colors, markers):
        partitions = np.array(results[key]['partitions'])
        delay = results[key]['delay']
        delay_mean = np.array(delay['mean'])
        delay_min = np.array(delay['min'])
        delay_max = np.array(delay['max'])
        plt.semilogx(partitions, delay_mean, color + marker, label=key)
        yerr = np.array([delay_mean - delay_min, delay_max - delay_mean])
        plt.errorbar(partitions, delay_mean, yerr=yerr, fmt='none', ecolor=color)

    x1, x2, y1, y2 = plt.axis()
    plt.axis([x1, x2, 0.95, 1.5])
    plt.setp(ax2.get_xticklabels(), fontsize=12, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=10)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()

    return

def load_delay_plots(parameters):
    """ Plot load and delay against problem size.

    Attempts to load results from disk and will skip any missing results.

    Args:
    parameters: The parameters for which to plot.
    """

    # List of directories with results
    #directories = ['./results/random/', './results/hybrid/', './results/block/']
    directories = ['./results/random/', './results/heuristic/']
    #colors = ['b', 'r', 'k']
    colors = ['b', 'r']
    markers = ['o', 'd']
    #keys = ['random', 'hybrid', 'block']
    keys = ['Random', 'Heuristic']

    # Setup somewhere to store the results we need
    results = dict()
    for par in parameters:
        for directory, key in zip(directories, keys):
            results[key] = dict()
            result = results[key]

            result['load'] = dict()
            result['load']['mean'] = list()
            result['load']['min'] = list()
            result['load']['max'] = list()

            result['delay'] = dict()
            result['delay']['mean'] = list()
            result['delay']['min'] = list()
            result['delay']['max'] = list()

            result['servers'] = list()

    # Load the results from disk
    for par in parameters:
        for directory, key in zip(directories, keys):
            try:
                df = pd.read_csv(directory + par.identifier() + '.csv')
            except FileNotFoundError:
                print('No data for', directory, par.identifier())
                continue

            result = results[key]

            load = df['load'] + par.num_multicasts()
            load /= par.num_source_rows * par.num_outputs * par.unpartitioned_load()
            result['load']['mean'].append(load.mean())
            result['load']['min'].append(load.min())
            result['load']['max'].append(load.max())

            delay = df['delay']
            delay /= par.computational_delay()
            result['delay']['mean'].append(delay.mean())
            result['delay']['min'].append(delay.min())
            result['delay']['max'].append(delay.max())

            result['servers'].append(par.num_servers)

    # Plot the load
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.grid(True, which='both')
    plt.ylabel('Comm. Load Increase', fontsize=15)
    #plt.xlabel('$K$', fontsize=15)
    plt.autoscale()

    for key, color, marker in zip(keys, colors, markers):
        servers = np.array(results[key]['servers'])
        load = results[key]['load']
        load_mean = np.array(load['mean'])
        load_min = np.array(load['min'])
        load_max = np.array(load['max'])
        plt.semilogx(servers, load_mean, color + marker, label=key)
        yerr = np.array([load_mean - load_min, load_max - load_mean])
        plt.errorbar(servers, load_mean, yerr=yerr, fmt='none', ecolor=color)

    plt.legend(numpoints=1)
    plt.setp(ax1.get_yticklabels(), fontsize=10)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Plot the delay
    ax2 = plt.subplot(212, sharex=ax1)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', weight='bold')
    #plt.rc('font', weight='bold')
    plt.grid(True, which='both')
    plt.ylabel('Comp. Delay Increase', fontsize=15)
    plt.xlabel('$K$', fontsize=15)
    plt.autoscale()

    for key, color, marker in zip(keys, colors, markers):
        servers = np.array(results[key]['servers'])
        delay = results[key]['delay']
        delay_mean = np.array(delay['mean'])
        delay_min = np.array(delay['min'])
        delay_max = np.array(delay['max'])
        plt.semilogx(servers, delay_mean, color + marker, label=key)
        yerr = np.array([delay_mean - delay_min, delay_max - delay_mean])
        plt.errorbar(servers, delay_mean, yerr=yerr, fmt='none', ecolor=color)

    #plt.legend()
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), fontsize=12, weight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=10)

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
    num_partitions = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30, 40, 50, 60,
                      75, 100, 120, 125, 150, 200, 250, 300, 375, 500, 600, 750,
                      1000, 1500, 3000]

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(rows_per_batch,
                                     num_servers,
                                     q,
                                     num_outputs,
                                     server_storage,
                                     partitions)
        parameters.append(par)

    return parameters

def main():
    """ Main examples function """

    ## Run simulations for various solvers and parameters
    # Load-delay parameters
    simulate(get_parameters_load_delay(), heuristicsolver.HeuristicSolver(), './results/', 1)
    simulate(get_parameters_load_delay(), randomsolver.RandomSolver(), './resultsm/', 100)

    # Partitioning parameters
    simulate(get_parameters_partitioning(), heuristicsolver.HeuristicSolver(), './results/', 1)
    simulate(get_parameters_partitioning(), randomsolver.RandomSolver(), './results/', 100)

    # Create the plots
    load_delay_plots(get_parameters_load_delay())
    partitioning_plots(get_parameters_partitioning())

    return

if __name__ == '__main__':
    main()
