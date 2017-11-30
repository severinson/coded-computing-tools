'''Script to create the plots for the journal paper

'''

import logging
import complexity
import numpy as np
import matplotlib.pyplot as plt
import stats
import model

from plot import get_parameters_partitioning, get_parameters_size, load_delay_plot
from simulation import Simulator
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver

def stats_plots():
    '''per server and overall delay distribution plots'''
    order = 18
    total = 27
    complexity = 1
    x = np.linspace(0, 12*complexity, 1000)

    # per server
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

    # plt.grid(True, which='both')
    plt.ylabel('$\mathsf{Probability\;Density}$', fontsize=24)
    plt.xlabel('$\mathsf{Time}$', fontsize=24)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    # plt.title('Delay per Server', fontsize=24)
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
    plt.savefig('./plots/itw/subtask_1.pdf')

    # Coded server runtime
    parameter = complexity * total / order
    rv = stats.Shiftexp(parameter)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='Coded')
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        fontsize=24,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.savefig('./plots/itw/subtask_2.pdf')

    # overall
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

    # plt.grid(True, which='both')
    plt.ylabel('$\mathsf{Probability\;Density}$', fontsize=24)
    plt.xlabel('$\mathsf{Time}$', fontsize=24)
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)
    # plt.title('Overall Delay', fontsize=24)
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
    plt.savefig('./plots/itw/overall_1.pdf')

    # Coded overall runtime
    parameter = complexity * total / order
    rv = stats.ShiftexpOrder(parameter, total, order)
    y = [rv.pdf(t) for t in x]
    plt.plot(x, y, color='k', linewidth=1)
    ax.fill_between(x, 0, y, color='b', alpha=0.7,
                    label='Coded')
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        fontsize=24,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.savefig('./plots/itw/overall_2.pdf')
    return

def main():
    '''Create plots for the ITW presentation.'''

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

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    # TODO: Partial
    rs_partitions.set_reduce_delay(function=lambda x:
                                   complexity.partitioned_reduce_delay(x, partitions=1))

    # Simulate size parameters
    heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    random_size = random_sim.simulate_parameter_list(size_parameters)
    rs_size = rs_sim.simulate_parameter_list(size_parameters)
    rs_size_decoding = rs_sim.simulate_parameter_list(size_parameters)

    # Include the reduce delay
    heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    # TODO: partial
    rs_size_decoding.set_reduce_delay(
        function=lambda x: complexity.partitioned_reduce_delay(x, partitions=1)
    )

    # Plots for presentation
    rs_plot_settings = {
        'label': r'RS excl. decoding',
        'color': 'c',
        'marker': '-o',
        'linewidth': 4,
        'size': 7}
    rs_decoding_plot_settings = {
        'label': r'RS',
        'color': 'k',
        'marker': 'd--',
        'linewidth': 4,
        'size': 7}
    heuristic_plot_settings = {
        'label': 'BDC, Heuristic',
        'color': 'r',
        'marker': '-H',
        'linewidth': 4,
        'size': 7}
    random_plot_settings = {
        'label': 'BDC, Random',
        'color': 'b',
        'marker': '-^',
        'linewidth': 4,
        'size': 8}

    # The unified scheme
    load_delay_plot(
        [rs_size],
        [rs_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size,
        show=False,
    )
    plt.savefig('./plots/itw/unified_1.pdf')

    load_delay_plot(
        [rs_size, rs_size_decoding],
        [rs_plot_settings, rs_decoding_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size,
        show=False,
    )
    plt.savefig('./plots/itw/unified_2.pdf')

    load_delay_plot(
        [rs_size, rs_size_decoding],
        [rs_plot_settings, rs_decoding_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size_decoding,
        show=False,
    )
    plt.savefig('./plots/itw/unified_3.pdf')

    # Numerical results, size
    load_delay_plot(
        [rs_size_decoding],
        [rs_decoding_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size_decoding,
        show=False,
    )
    plt.savefig('./plots/itw/size_1.pdf')

    load_delay_plot(
        [rs_size_decoding, random_size],
        [rs_decoding_plot_settings, random_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size_decoding,
        show=False,
    )
    plt.savefig('./plots/itw/size_2.pdf')

    load_delay_plot(
        [rs_size_decoding, random_size, heuristic_size],
        [rs_decoding_plot_settings, random_plot_settings, heuristic_plot_settings],
        'servers',
        xlabel='Servers $K$',
        normalize=rs_size_decoding,
        show=False,
    )
    plt.savefig('./plots/itw/size_3.pdf')

    # Numerical results, partitions
    load_delay_plot(
        [rs_partitions],
        [rs_decoding_plot_settings],
        'partitions',
        xlabel='Partitions $T$',
        normalize=rs_partitions,
        show=False,
    )
    plt.savefig('./plots/itw/partitions_1.pdf')

    load_delay_plot(
        [rs_partitions, random_partitions],
        [rs_decoding_plot_settings, random_plot_settings],
        'partitions',
        xlabel='Partitions $T$',
        normalize=rs_partitions,
        show=False,
    )
    plt.savefig('./plots/itw/partitions_2.pdf')

    load_delay_plot(
        [rs_partitions, random_partitions, heuristic_partitions],
        [rs_decoding_plot_settings, random_plot_settings, heuristic_plot_settings],
        'partitions',
        xlabel='Partitions $T$',
        normalize=rs_partitions,
        show=False,
    )
    plt.savefig('./plots/itw/partitions_3.pdf')
    return

def get_parameters_example():
    rows_per_batch = 2
    num_servers = 4
    q = 2
    num_outputs = q
    server_storage = 1
    num_partitions = 6
    parameters = model.SystemParameters(
        rows_per_batch=rows_per_batch,
        num_servers=num_servers,
        q=q,
        num_outputs=num_outputs,
        server_storage=server_storage,
        num_partitions=num_partitions,
    )
    print(parameters)
    return parameters

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # stats_plots()
    # get_parameters_example()
