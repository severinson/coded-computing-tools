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
import numpy as np
import model
import stats
from simulation import Simulator
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
import complexity
from solvers.assignmentloader import AssignmentLoader
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from assignments.cached import CachedAssignment
from plot import load_delay_plot, complexity_plot

# Set default plot fonts
rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)

def example1_plots():
    x = np.linspace(0, 12, 1000)
    order = 18
    total = 27

    _, ax = plt.subplots()
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)

    # Uncoded server runtime
    parameter = 1
    rv = stats.Shiftexp(parameter)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='r', alpha=0.7,
                    label='UC')

    # Coded server runtime
    parameter = total / order
    rv = stats.Shiftexp(parameter)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='SC')

    plt.grid(True, which='both')
    plt.ylabel('Runtime Probability Density', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    # plt.title('Server Runtime')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.legend(numpoints=1, fontsize=22, loc='best')
    plt.tight_layout()
    plt.savefig('./plots/report/example1_subtask.pgf')

    _, ax = plt.subplots()
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)

    # Uncoded overall runtime
    parameter = 1
    rv = stats.ShiftexpOrder(parameter, total, total)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='r', alpha=0.7,
                    label='UC')

    # Coded overall runtime
    parameter = total / order
    rv = stats.ShiftexpOrder(parameter, total, order)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='SC')

    plt.grid(True, which='both')
    plt.ylabel('Runtime Probability Density', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    # plt.title('Overall Runtime')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.legend(numpoints=1, fontsize=22, loc='best')

    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.savefig('./plots/report/example1_map.pgf')
    # plt.subplots_adjust(wspace=0, hspace=0.2)

    plt.show()
    return

def shiftexp_plot():
    '''Plot a shifted exponential for various parameters.'''
    parameters = [1, 2, 3]
    colors = ['r', 'g', 'b']
    patterns = ['/', '*', '+']
    _, ax = plt.subplots(figsize=(8,6))
    x = np.linspace(0, 16, 1000)
    for p, color, pattern in zip(parameters, colors, patterns):
        rv = stats.Shiftexp(p)
        y = [rv.cdf(t) for t in x]
        plt.plot(x, y, color='k', linewidth=1)
        ax.fill_between(x, 0, y, color=color, alpha=0.7,
                        linewidth='2', linestyle='-', edgecolor='k',
                        label='$\sigma=' + str(p) + '$')

    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.grid(True, which='both')
    plt.ylabel('Completion Probability', fontsize=18)
    plt.xlabel('$t$', fontsize=18)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.legend(numpoints=1, fontsize=18, loc='best')
    plt.savefig('./plots/report/stats/shiftexp.pgf')
    # plt.show()
    return

def stats_plots():
    order = 18
    total = 27
    complexity = 1
    x = np.linspace(0, 12*complexity, 1000)
    labels = ['$0$', '$\sigma$'] + ['$' + str(i) + '\sigma$' for i in range(2, 13)]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    _, ax = plt.subplots()

    # Uncoded server runtime
    parameter = complexity
    rv = stats.Shiftexp(parameter)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='r', alpha=0.7,
                    label='Uncoded')

    # Coded server runtime
    parameter = complexity * total / order
    rv = stats.Shiftexp(parameter)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='Coded')

    plt.grid(True, which='both')
    plt.ylabel('Delay Probability Density', fontsize=28)
    plt.xlabel('Time', fontsize=28)
    plt.xticks(range(13), labels)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    plt.title('Delay per Server', fontsize=24)
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        fontsize=24,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.tight_layout()
    plt.savefig('./plots/itw/subtask.pgf')

    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    _, ax = plt.subplots()

    # Uncoded overall runtime
    parameter = complexity
    rv = stats.ShiftexpOrder(parameter, total, total)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='r', alpha=0.7,
                    label='Uncoded')

    # Coded overall runtime
    parameter = complexity * total / order
    rv = stats.ShiftexpOrder(parameter, total, order)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='Coded')

    plt.grid(True, which='both')
    plt.ylabel('Delay Probability Density', fontsize=28)
    plt.xlabel('Time', fontsize=28)
    plt.xticks(range(13), labels)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    plt.title('Overall Delay', fontsize=24)
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        fontsize=24,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )

    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.savefig('./plots/itw/overall.pgf')
    # plt.subplots_adjust(wspace=0, hspace=0.2)

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
    '''Create and save the plots for the report.'''

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

    stragglerc_sim = Simulator(solver=None, assignments=1,
                               parameter_eval=analytic.stragglerc_performance,
                               directory='./results/Stragglerc/')

    hybrid_solver = AssignmentLoader(directory='./results/Hybrid/assignments/')
    hybrid_sim = Simulator(solver=hybrid_solver, assignments=1,
                           assignment_eval=sample_1000,
                           assignment_type=CachedAssignment,
                           directory='./results/Hybrid/')

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    hybrid_partitions = hybrid_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)
    cmapred_partitions = cmapred_sim.simulate_parameter_list(partition_parameters)
    stragglerc_partitions = stragglerc_sim.simulate_parameter_list(partition_parameters)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    hybrid_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(
        function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    uncoded_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_uncoded(enable=True)
    cmapred_partitions.set_reduce_delay(function=lambda x: 0)
    cmapred_partitions.set_cmapred(enable=True)
    stragglerc_partitions.set_reduce_delay(function=complexity.stragglerc_reduce_delay)
    stragglerc_partitions.set_stragglerc(enable=True)

    # Include the encoding delay
    heuristic_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    random_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    hybrid_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    rs_partitions.set_encode_delay(
        function=lambda x: complexity.partitioned_encode_delay(x, partitions=1))
    uncoded_partitions.set_encode_delay(function=lambda x: 0)
    cmapred_partitions.set_encode_delay(function=lambda x: 0)
    stragglerc_partitions.set_encode_delay(function=complexity.stragglerc_encode_delay)

    # Simulate size parameters
    heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    random_size = random_sim.simulate_parameter_list(size_parameters)
    rs_size = rs_sim.simulate_parameter_list(size_parameters)
    uncoded_size = uncoded_sim.simulate_parameter_list(size_parameters)
    cmapred_size = cmapred_sim.simulate_parameter_list(size_parameters)
    stragglerc_size = stragglerc_sim.simulate_parameter_list(size_parameters)

    # Include the encoding delay
    heuristic_size.set_encode_delay(function=complexity.partitioned_encode_delay)
    random_size.set_encode_delay(function=complexity.partitioned_encode_delay)
    rs_size.set_encode_delay(
        function=lambda x: complexity.partitioned_encode_delay(x, partitions=1))
    uncoded_size.set_encode_delay(function=lambda x: 0)
    cmapred_size.set_encode_delay(function=lambda x: 0)
    stragglerc_size.set_encode_delay(function=complexity.stragglerc_encode_delay)

    # Include the reduce delay
    heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_size.set_reduce_delay(
        function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1))
    uncoded_size.set_reduce_delay(function=lambda x: 0)
    uncoded_size.set_uncoded(enable=True)
    cmapred_size.set_reduce_delay(function=lambda x: 0)
    cmapred_size.set_cmapred(enable=True)
    stragglerc_size.set_reduce_delay(function=complexity.stragglerc_reduce_delay)
    stragglerc_size.set_stragglerc(enable=True)

    # Plots for presentation
    rs_plot_settings = {
        'label': r'Unified',
        'color': 'r',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}
    cmapred_plot_settings = {
        'label': r'CMR',
        'color': 'g',
        'marker': 'v',
        'linewidth': 2,
        'size': 7}
    stragglerc_plot_settings = {
        'label': r'SC',
        'color': 'k',
        'marker': '--H',
        'linewidth': 2,
        'size': 7}
    heuristic_plot_settings = {
        'label': 'BDC, Heuristic',
        'color': 'b',
        'marker': 's',
        'linewidth': 2,
        'size': 7}
    random_plot_settings = {
        'label': 'BDC, Random',
        'color': 'g',
        'marker': '^',
        'linewidth': 2,
        'size': 8}
    hybrid_plot_settings = {
        'label': 'BDC, Hybrid',
        'color': 'r',
        'marker': 'H',
        'linewidth': 2,
        'size': 6}

    # Compare coded mapreduce, stragglerc, and unified with BDC
    # load_delay_plot([rs_size, stragglerc_size, cmapred_size, heuristic_size],
    #                 [rs_plot_settings, stragglerc_plot_settings,
    #                  cmapred_plot_settings, heuristic_plot_settings],
    #                 'servers', xlabel='$K$', normalize=uncoded_size, legend='delay')
    # # plt.ylim(0.9, 1.1)
    # plt.savefig('./plots/report/numerical/size.pgf')
    # plt.savefig('./plots/report/numerical/size.pdf')

    # load_delay_plot([rs_partitions, stragglerc_partitions, cmapred_partitions, heuristic_partitions],
    #                 [rs_plot_settings, stragglerc_plot_settings,
    #                  cmapred_plot_settings, heuristic_plot_settings],
    #                 'partitions', xlabel='$T$', normalize=uncoded_partitions, legend='load')
    # plt.savefig('./plots/report/numerical/partitions.pgf')
    # plt.savefig('./plots/report/numerical/partitions.pdf')

    # Comparing the solvers
    heuristic_plot_settings = {
        'label': 'Heuristic',
        'color': 'b',
        'marker': 's',
        'linewidth': 2,
        'size': 7}
    random_plot_settings = {
        'label': 'Random',
        'color': 'g',
        'marker': '^',
        'linewidth': 2,
        'size': 8}
    hybrid_plot_settings = {
        'label': 'Hybrid',
        'color': 'r',
        'marker': 'H',
        'linewidth': 2,
        'size': 6}

    # load_delay_plot([heuristic_size, random_size],
    #                 [heuristic_plot_settings, random_plot_settings],
    #                 'servers', xlabel='$K$', normalize=uncoded_size, legend='delay')
    # plt.savefig('./plots/report/numerical/solvers_size.pgf')
    # plt.savefig('./plots/report/numerical/solvers_size.pdf')

    # load_delay_plot([heuristic_partitions, random_partitions, hybrid_partitions],
    #                 [heuristic_plot_settings, random_plot_settings, hybrid_plot_settings],
    #                 'partitions', xlabel='$T$', normalize=uncoded_partitions, legend='load')
    # plt.savefig('./plots/report/numerical/solvers_partitions.pgf')
    # plt.savefig('./plots/report/numerical/solvers_partitions.pdf')

    # Complexity BDC vs. Reed-Solomon and straggler coding
    # Encoding delay, size
    complexity_plot([rs_size, stragglerc_size],
                    [rs_plot_settings, stragglerc_plot_settings],
                    'servers', xlabel='$K$', normalize=heuristic_size,
                    phase='encode')
    plt.savefig('./plots/report/encode_size.pgf')
    plt.savefig('./plots/report/encode_size.pdf')

    # Decoding delay, size
    complexity_plot([rs_size, stragglerc_size],
                    [rs_plot_settings, stragglerc_plot_settings],
                    'servers', xlabel='$K$', normalize=heuristic_size,
                    phase='reduce')
    plt.savefig('./plots/report/reduce_size.pgf')
    plt.savefig('./plots/report/reduce_size.pdf')

    # Encoding delay, partitions
    complexity_plot([rs_partitions, stragglerc_partitions],
                    [rs_plot_settings, stragglerc_plot_settings],
                    'partitions', xlabel='$T$', normalize=heuristic_partitions,
                    phase='encode')
    plt.savefig('./plots/report/encode_partitions.pgf')
    plt.savefig('./plots/report/encode_partitions.pdf')

    # Decoding delay, partitions
    complexity_plot([rs_partitions, stragglerc_partitions],
                    [rs_plot_settings, stragglerc_plot_settings],
                    'partitions', xlabel='$T$', normalize=heuristic_partitions,
                    phase='reduce')
    plt.savefig('./plots/report/reduce_partitions.pgf')
    plt.savefig('./plots/report/reduce_partitions.pdf')

    return

    # The unified scheme
    load_delay_plot([rs_size],
                    [rs_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/unified_1.pdf')

    load_delay_plot([rs_size, cmapred_size],
                    [rs_plot_settings, cmapred_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/unified_2.pdf')

    load_delay_plot([rs_size, cmapred_size],
                    [rs_plot_settings, cmapred_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/unified_3.pdf')

    # Numerical results, size
    load_delay_plot([rs_size],
                    [rs_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/size_1.pdf')

    load_delay_plot([rs_size, random_size],
                    [rs_plot_settings, random_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/size_2.pdf')

    load_delay_plot([rs_size, random_size, heuristic_size],
                    [rs_plot_settings, random_plot_settings, heuristic_plot_settings],
                    'servers', xlabel='Servers $K$', normalize=uncoded_size)
    plt.savefig('./plots/report/size_3.pdf')

    # Numerical results, partitions
    load_delay_plot([rs_partitions],
                    [rs_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/report/partitions_1.pdf')

    load_delay_plot([rs_partitions, random_partitions],
                    [rs_plot_settings, random_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/report/partitions_2.pdf')

    load_delay_plot([rs_partitions, random_partitions, heuristic_partitions],
                    [rs_plot_settings, random_plot_settings, heuristic_plot_settings],
                    'partitions', xlabel='Partitions $T$', normalize=uncoded_partitions)
    plt.savefig('./plots/report/partitions_3.pdf')
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # shiftexp_plot()
    stats_plots()
    # main()
    # example1_plots()
