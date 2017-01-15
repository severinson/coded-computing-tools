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

# This is file contains some usage examples.

import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model
import solvers

# Evaluate the speed of the greedy solver
def greedyRuntime():
    rows_per_batch = [2, 20, 40, 80, 160]
    results = dict()
    for r in rows_per_batch:
        p = model.SystemParameters(r, # Rows per batch
                                   6, # Number of servers (K)
                                   4, # Servers to wait for (q)
                                   4, # Outputs (N)
                                   1/2, # Server storage (\mu)
                                   1) # Partitions (T)
        main.greedyPerformance(p, [1], n=10)

# Evaluate the performance for varying size matrices using the hybrid
# solvers.
def ex1():
    num_servers = 6
    q = 4
    server_storage = 1/2
    #rows_per_batch = [2, 4, 8, 16, 32, 64, 128]
    rows_per_batch = [128];
    #partitions = [10, 20, 40, 80, 160, 320, 640]
    partitions = [640];
    parameters = list()
    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i],
                                   num_servers,
                                   q,
                                   q,
                                   server_storage,
                                   partitions[i])
        parameters.append(p)

    results = main.performanceEval(parameters)
    return results

# Evaluate the performance of random assignments for varying size
# matrices.
def ex2():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [256, 512, 1024, 2048, 4096, 8192, 16384]
    partitions = [1280, 2560, 5120, 10240, 20480, 40960, 81920]

    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i], num_servers, q, q, server_storage, partitions[i])
        print(p)

        df = list()
        n = 100
        for j in range(n):
            X, A = solvers.assignmentRandom(p)
            avg_load, avg_missed_delay = solvers.objective_function_sampled(p, X, A)

            result = dict()
            result['sampled'] = avg_load
            df.append(result)

        df = pd.Series(df)
        df.to_csv(p.identifier() + '.csv')

    return

# Compare the exhaustive and sampled objective functions
def ex3():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    partitions = [2, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920]
    n = 100

    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i], num_servers, q, q, server_storage, partitions[i])

        # Try to load the results from disk
        try:
            pd.read_csv('./results/' + p.identifier() + '.csv')
            continue

        # Generate the data if we couldn't find it
        except FileNotFoundError:
            print('Running simulations for:')
            print(p)

            df = list()
            for j in range(n):
                X, A = solvers.assignmentRandom(p)
                avg_load, avg_missed_delay = solvers.objective_function_sampled(p, X, A)

                result = dict()
                result['sampled'] = avg_load
                df.append(result)

            df = pd.DataFrame(df)
            df.to_csv('./results/' + p.identifier() + '.csv')

    return

def ex4():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    partitions = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    results = list()

    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i], num_servers, q, q, server_storage, partitions[i])
        df = pd.read_csv(p.identifier() + '.csv')
        results = list()

        rows = int(df.size / 2)
        for j in range(rows):
            r = eval(df.iloc[j][1])
            results.append(r)

        df = pd.DataFrame(results)
        df.to_csv(p.identifier() + '.csv')

    return

def ex5():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    partitions = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920]
    results = list()

    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i], num_servers, q, q, server_storage, partitions[i])
        df = pd.read_csv('./results/' + p.identifier() + '.csv')
        load = (df['sampled'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogx([p.num_source_rows for x in range(len(load))], load, 'b.')

        '''
        if 'exhaustive' in df:
            load = df['exhaustive'] / p.num_source_rows / p.num_outputs / p.unpartitioned_load()
            plt.semilogx([p.num_source_rows for x in range(len(load))], load, 'b.')
        '''

        plt.grid()
        plt.ylabel('Communication Load Increase', fontsize=15)
        plt.xlabel('$m$', fontsize=15)

    plt.show()
    return

def ex6():
    num_servers = [60, 600, 6000]
    q = [40, 400, 4000]
    server_storage = [1/x for x in q]
    num_outputs = q
    rows_per_batch = 2
    partitions = [10, 100, 1000]
    n = 100

    for i in range(len(q)):
        p = model.SystemParameters(rows_per_batch, num_servers[i], q[i], num_outputs[i], server_storage[i], partitions[i])

        # Try to load the results from disk
        try:
            pd.read_csv('./results/' + p.identifier() + '.csv')
            continue

        # Generate the data if we couldn't find it
        except FileNotFoundError:
            print('Running simulations for:')
            print(p)

            df = list()
            for j in range(n):
                X, A = solvers.assignmentRandom(p)
                avg_load, avg_missed_delay = solvers.objective_function_sampled(p, X, A)

                result = dict()
                result['sampled'] = avg_load
                result['delay'] = avg_missed_delay
                df.append(result)

            df = pd.DataFrame(df)
            df.to_csv(p.identifier() + '.csv')

    return

# Histograms
def ex7():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [2]#, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    partitions = [10]#, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]

    for i in range(len(rows_per_batch)):
        p = model.SystemParameters(rows_per_batch[i], num_servers, q, q, server_storage, partitions[i])
        df = pd.read_csv('./results/' + p.identifier() + '.csv')

        load = (df['sampled'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()
        fig = plt.figure()
        plt.hist(load, bins=20, normed=True)
        plt.grid()
        plt.xlabel('Communication Load Increase')
        plt.ylabel('Probability Density')

        load = (df['exhaustive'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()
        fig = plt.figure()
        plt.hist(load, bins=20, normed=True)
        plt.grid()
        plt.xlabel('Communication Load Increase')
        plt.ylabel('Probability Density')

    plt.show()
    return

def ex8():
    num_servers = [60, 600, 6000]
    q = [40, 400, 4000]
    server_storage = [1/x for x in q]
    num_outputs = q
    rows_per_batch = 2
    partitions = [10, 100, 1000]
    n = 100

    for i in range(len(q)):
        p = model.SystemParameters(rows_per_batch, num_servers[i], q[i], num_outputs[i], server_storage[i], partitions[i])
        df = pd.read_csv('./results/' + p.identifier() + '.csv')
        load = (df['sampled'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.semilogx([p.num_servers for x in range(len(load))], load, 'b.')
        plt.grid()
        plt.ylabel('Communication Load Increase', fontsize=15)
        plt.xlabel('$K$', fontsize=15)

    plt.show()
    return

def ex9():
    num_servers = [60, 600]
    q = [40, 400]
    server_storage = [1/x for x in q]
    num_outputs = q
    rows_per_batch = 2
    partitions = [100, 1000]
    parameters = [model.SystemParameters(rows_per_batch, x[0], x[1], x[2], x[3], x[4])
                  for x in zip(num_servers, q, num_outputs, server_storage, partitions)]

    directory = './tmp/'
    simulations(parameters, directory=directory)
    plots(parameters, directory=directory)

def ex10():
    num_servers = 6
    q = 4
    server_storage = 1/2
    rows_per_batch = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    partitions = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    parameters = [model.SystemParameters(x[0], num_servers, q, q, server_storage, x[1]) for x in zip(rows_per_batch, partitions)]

    directory = './tmp/'
    simulations(parameters, directory=directory)
    plots(parameters, directory=directory)

def ex11():
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    min_num_servers = 3
    parameters = list()
    for i in range(7):
        p = model.SystemParameters.parameters_from_rows(rows_per_server,
                                                        rows_per_partition,
                                                        min_num_servers,
                                                        code_rate,
                                                        muq)
        min_num_servers = p.num_servers
        parameters.append(p)
        print(p)

    directory = './results/heuristic/'
    #simulations(parameters, solvers.assignment_heuristic, n=100, directory=directory)

    par = parameters[-1]
    simulations([par], solvers.assignment_heuristic, n=100, directory=directory)

    directories = ['./results/random/', './results/hybrid/', './results/heuristic/']
    plots(parameters, directories=directories)

# Histogram of the sampled objective functions
def ex12():
    rows_per_batch = 16
    num_servers = 126
    q = 84
    server_storage = 2 / q
    num_outputs = q
    partitions = 8400
    p = model.SystemParameters(rows_per_batch, num_servers, q, num_outputs, server_storage, partitions)
    directory = './histogram_repeated/'

    # Try to load the results from disk
    try:
        df = pd.read_csv(directory + p.identifier() + '.csv')

    # Otherwise run the simulations
    except FileNotFoundError:
        print('Running simulations for:')
        print(p)

        X, A = solvers.assignmentRandom(p)
        assert solvers.is_valid(p, X)

        n = 1000
        df = list()
        for i in range(n):

            # Evaluate it
            avg_load, avg_delay = solvers.sampledObjectiveFunction(X, A, p, n=1)
            print(avg_load, avg_delay)

            # Store the results
            result = dict()
            result['load'] = avg_load
            result['delay'] = avg_delay
            df.append(result)

            # Write a dot to show that progress is being made
            sys.stdout.write('.')
            sys.stdout.flush()

        print('')

        # Create a pandas dataframe and write it to disk
        df = pd.DataFrame(df)
        df.to_csv(directory + p.identifier() + '.csv')

    # Histogram of the load
    fig = plt.figure()
    load = (df['load'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.hist(load, bins=20, normed=True)
    plt.hist(load[0:100], bins=20, normed=True)
    plt.grid(True, which='both')
    #plt.ylabel('Communication Load Increase', fontsize=15)
    #plt.xlabel('$K$', fontsize=15)

    #plt.plot(load.mean(), 0, 'b*')
    #plt.plot(load[0:100].mean(), 0, 'r*')

    # Histogram of the delay
    fig = plt.figure()
    delay = df['delay'] / p.q
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.hist(delay, bins=20, normed=True)
    plt.hist(delay[0:100], bins=20, normed=True)
    plt.grid(True, which='both')
    #plt.ylabel('Computational Delay Increase', fontsize=15)
    #plt.xlabel('$K$', fontsize=15)

    plt.show()
    return

def rename_files():
    """ Rename files to conform to the new identifier standard. """
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    min_num_servers = 3
    parameters = list()
    old_directory = './tmp/'
    directory = './results/'
    for i in range(7):
        p = model.SystemParameters.parameters_from_rows(rows_per_server,
                                                        rows_per_partition,
                                                        min_num_servers,
                                                        code_rate,
                                                        muq)
        min_num_servers = p.num_servers
        df = pd.read_csv(old_directory + p.old_identifier() + '.csv')
        df.to_csv(directory + p.identifier() + '.csv')
        print(p)

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
                assignment = solver(par, verbose=verbose)
                assert model.is_valid(par, assignment.assignment_matrix, verbose=True)

                # Evaluate it
                avg_load, avg_delay = solvers.objective_function_sampled(par,
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

            load = (df['load'] + par.num_multicasts()) / par.num_source_rows / par.num_outputs / par.unpartitioned_load()
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

    x1,x2,y1,y2 = plt.axis()
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

    x1,x2,y1,y2 = plt.axis()
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
    width = [10, 10, 7]
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

            load = (df['load'] + par.num_multicasts()) / par.num_source_rows / par.num_outputs / par.unpartitioned_load()
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

def load_delay_scatter_plots():
    """ Scatter plots of the  load and delay for the parameters returned by get_parameters().

    Attempts to load results from disk and will skip any missing results.
    """

    for par in parameters:
        for directory, color, w in zip(directories, colors, width):
            try:
                df = pd.read_csv(directory + par.identifier() + '.csv')
            except FileNotFoundError:
                continue

            load = (df['load'] + par.num_multicasts()) / par.num_source_rows / par.num_outputs / par.unpartitioned_load()
            load_mean = load.mean()
            #plt.semilogx([par.num_servers for x in range(len(load))], load, color + '.')
            plt.semilogx(par.num_servers, load_mean, color + '.', label='YO')
            yerr = np.array([[load_mean - load.min()], [load.max() - load_mean]])
            plt.errorbar(par.num_servers, load_mean, yerr=yerr, ecolor=color, capsize=w)

    #plt.legend(handles, ['Random', 'Heuristic + random', 'Hybrid'])
    plt.legend()

    # Plot the delay
    fig = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.grid(True, which='both')
    plt.ylabel('Computational Delay Increase', fontsize=15)
    plt.xlabel('$K$', fontsize=15)
    plt.autoscale()
    plt.legend(['Random', 'Heuristic + random', 'Hybrid'])

    for par in parameters:
        for directory, color, w in zip(directories, colors, width):
            try:
                df = pd.read_csv(directory + par.identifier() + '.csv')
            except FileNotFoundError:
                continue

            delay = df['delay'] / par.q
            delay_mean = delay.mean()

            #plt.semilogx([par.num_servers for x in range(len(delay))], delay, marker)
            plt.semilogx(par.num_servers, delay_mean, color + '.')
            yerr = np.array([[delay_mean - delay.min()], [delay.max() - delay_mean]])
            plt.errorbar(par.num_servers, delay_mean, yerr=yerr, ecolor=color, capsize=w)
    plt.show()

def plots(parameters, directories=['./results/']):
    fig = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.grid(True, which='both')
    plt.ylabel('Communication Load Increase', fontsize=15)
    plt.xlabel('$K$', fontsize=15)

    for p in parameters:
        for directory in directories:
            try:
                df = pd.read_csv(directory + p.identifier() + '.csv')
            except FileNotFoundError:
                continue

            if 'load' not in df:
                continue

            load = (df['load'] + p.num_multicasts()) / p.num_source_rows / p.num_outputs / p.unpartitioned_load()
            plt.semilogx([p.num_servers for x in range(len(load))], load, '.')

    fig = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.grid(True, which='both')
    plt.ylabel('Computational Delay Increase', fontsize=15)
    plt.xlabel('$K$', fontsize=15)

    for p in parameters:
        for directory in directories:
            try:
                df = pd.read_csv(directory + p.identifier() + '.csv')
            except FileNotFoundError:
                continue

            if 'delay' not in df:
                continue

            delay = df['delay'] / p.q
            plt.semilogx([p.num_servers for x in range(len(delay))], delay, '.')

    plt.show()

def get_parameters_load_delay():
    """ Get a list of parameters. """

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
    """ Get a list of parameters. """

    rows_per_batch = 250
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/3
    num_partitions = [2,3,4,5,6,8,10,12,15,20,24,25,30,40,50,60,75,100,120,125,150,200,250,300,375,500,600,750,1000,1500,3000]

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(rows_per_batch, num_servers, q, num_outputs, server_storage, partitions)
        parameters.append(par)

    return parameters

def main():
    """ Main examples function """

    for par in get_parameters_load_delay():
        print(par.num_source_rows / par.num_partitions)

    return
    parameters = get_parameters_load_delay()
    #parameters = get_parameters_partitioning()

    # Run simulations for various solvers
    simulate(parameters, solvers.assignment_block, './results/heuristic/', 1)
    #simulate(parameters, solvers.assignment_random, './results/random/', 100)

    load_delay_plots(get_parameters_load_delay())
    partitioning_plots(get_parameters_partitioning())
    return

if __name__ == '__main__':
    main()
