'''Script to create the plots for the journal paper

'''

import logging
import complexity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stats
import rateless

from functools import partial
from plot import get_parameters_size, load_delay_plot
from plot import get_parameters_partitioning, get_parameters_partitioning_2
from simulation import Simulator
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver

def main():
    '''Create plots for the ITW presentation.'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning_2()
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

    lt_partitions = [rateless.evaluate(
        partition_parameters[0],
        target_overhead=1.3,
        target_failure_probability=0.5,
    )] * len(partition_parameters)
    lt_partitions = pd.DataFrame(lt_partitions)
    lt_partitions['partitions'] = [parameters.num_partitions for parameters in partition_parameters]

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)
    cmapred_partitions = cmapred_sim.simulate_parameter_list(partition_parameters)
    stragglerc_partitions = stragglerc_sim.simulate_parameter_list(partition_parameters)

    # include encoding delay
    heuristic_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    random_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    rs_partitions.set_encode_delay(
        function=partial(complexity.partitioned_encode_delay, partitions=1)
    )

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(
        function=partial(complexity.partitioned_reduce_delay, partitions=1)
    )
    uncoded_partitions.set_reduce_delay(function=lambda x: 0)
    uncoded_partitions.set_uncoded(enable=True)
    cmapred_partitions.set_reduce_delay(function=lambda x: 0)
    cmapred_partitions.set_cmapred(enable=True)
    stragglerc_partitions.set_reduce_delay(function=complexity.stragglerc_reduce_delay)
    stragglerc_partitions.set_stragglerc(enable=True)

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
        'color': 'g',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}
    rs_decoding_plot_settings = {
        'label': r'RS',
        'color': 'k',
        'marker': 'd--',
        'linewidth': 2,
        'size': 7}
    heuristic_plot_settings = {
        'label': 'BDC, Heuristic',
        'color': 'r',
        'marker': '-H',
        'linewidth': 2,
        'size': 7}
    random_plot_settings = {
        'label': 'BDC, Random',
        'color': 'b',
        'marker': '-^',
        'linewidth': 2,
        'size': 8}
    lt_plot_settings = {
        'label': 'LT',
        'color': 'c',
        'marker': 'v',
        'linewidth': 3,
        'size': 8
    }

    # # The unified scheme
    # load_delay_plot(
    #     [rs_size],
    #     [rs_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/unified_1.pdf')

    # load_delay_plot(
    #     [rs_size, rs_size_decoding],
    #     [rs_plot_settings, rs_decoding_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/unified_2.pdf')

    # load_delay_plot(
    #     [rs_size, rs_size_decoding],
    #     [rs_plot_settings, rs_decoding_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size_decoding,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/unified_3.pdf')

    # # Numerical results, size
    # load_delay_plot(
    #     [rs_size, rs_size_decoding],
    #     [rs_plot_settings, rs_decoding_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size_decoding,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/size_1.pdf')

    # load_delay_plot(
    #     [rs_size, rs_size_decoding, random_size],
    #     [rs_plot_settings, rs_decoding_plot_settings, random_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size_decoding,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/size_2.pdf')

    # load_delay_plot(
    #     [rs_size, rs_size_decoding, random_size, heuristic_size],
    #     [rs_plot_settings, rs_decoding_plot_settings, random_plot_settings, heuristic_plot_settings],
    #     'servers',
    #     xlabel='$\mathsf{Servers}\;K$',
    #     normalize=rs_size_decoding,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/size_3.pdf')

    # # Numerical results, partitions
    # load_delay_plot(
    #     [rs_partitions],
    #     [rs_decoding_plot_settings],
    #     'partitions',
    #     xlabel='$\mathsf{Partitions}\;T$',
    #     normalize=rs_partitions,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/partitions_1.pdf')

    # load_delay_plot(
    #     [rs_partitions, random_partitions],
    #     [rs_decoding_plot_settings, random_plot_settings],
    #     'partitions',
    #     xlabel='$\mathsf{Partitions}\;T$',
    #     normalize=rs_partitions,
    #     show=False,
    # )
    # plt.savefig('./plots/itw/partitions_2.pdf')

    load_delay_plot(
        [heuristic_partitions, random_partitions, lt_partitions],
        [heuristic_plot_settings, random_plot_settings, lt_plot_settings],
        'partitions',
        xlabel='$\mathsf{Partitions}\;T$',
        normalize=uncoded_partitions,
        show=False,
    )
    plt.savefig('./plots/journal/partitions_lt.pdf')
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
