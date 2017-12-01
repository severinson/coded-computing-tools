'''Script to create the plots for the journal paper

'''

import logging
import complexity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stats
import rateless
import plot

from functools import partial
from plot import get_parameters_size, load_delay_plot
from plot import get_parameters_partitioning, get_parameters_partitioning_2
from plot import get_parameters_N
from simulation import Simulator
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

def N_n_ratio_plots():
    parameters = get_parameters_N()

    # simulate lt code performance
    target_overhead = 1.3
    lt = [rateless.evaluate(
        p,
        target_overhead=target_overhead,
        target_failure_probability=1e-3,
    ) for p in parameters]
    lt = pd.DataFrame(lt)

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Setup the simulators
    heuristic_sim = Simulator(
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
        directory='./results/Heuristic/',
    )

    uncoded_sim = Simulator(
        solver=None, assignments=1,
        parameter_eval=analytic.uncoded_performance,
        directory='./results/Uncoded/',
    )

    # run simulations
    uncoded = uncoded_sim.simulate_parameter_list(parameters)
    heuristic = heuristic_sim.simulate_parameter_list(parameters)

    # include encoding/decoding delay
    heuristic.set_encode_delay(function=complexity.partitioned_encode_delay)
    heuristic.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    uncoded_partitions.set_uncoded(enable=True)

    # plot settings
    settings_lt = {
        'label': 'LT tfp=1e-1',
        'color': 'g',
        'marker': '-o',
        'linewidth': 2,
        'size': 7}
    settings_heuristic = {
        'label': 'BDC heuristic',
        'color': 'k',
        'marker': 'x--',
        'linewidth': 4,
        'size': 8
    }

    plot.encode_decode_plot(
        [lt, heuristic],
        [settings_lt, settings_heuristic],
        'num_columns',
        xlabel='$\mathsf{Columns}\;n$',
        normalize=heuristic,
        show=False,
    )
    # plt.show()

    load_delay_plot(
        [lt, heuristic],
        [settings_lt, settings_heuristic],
        'num_columns',
        xlabel='$\mathsf{Columns}\;n$',
        normalize=uncoded,
        show=False,
    )
    plt.savefig('./plots/journal/N_n_ratio_3.png')
    plt.show()

def lt_plots():

    # get system parameters
    # parameter_list = plot.get_parameters_size_2()[:-4]
    parameter_list = plot.get_parameters_N()

    lt_2_1 = [rateless.evaluate(
        parameters,
        target_overhead=1.2,
        target_failure_probability=1e-1,
    ) for parameters in parameter_list]
    lt_2_1 = pd.DataFrame(lt_2_1)
    lt_2_1['servers'] = [parameters.num_servers
                         for parameters in parameter_list]

    lt_2_3 = [rateless.evaluate(
        parameters,
        target_overhead=1.2,
        target_failure_probability=1e-3,
    ) for parameters in parameter_list]
    lt_2_3 = pd.DataFrame(lt_2_3)
    lt_2_3['servers'] = [parameters.num_servers
                         for parameters in parameter_list]

    lt_3_1 = [rateless.evaluate(
        parameters,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    ) for parameters in parameter_list]
    lt_3_1 = pd.DataFrame(lt_3_1)
    lt_3_1['servers'] = [parameters.num_servers
                         for parameters in parameter_list]

    lt_3_3 = [rateless.evaluate(
        parameters,
        target_overhead=1.3,
        target_failure_probability=1e-3,
    ) for parameters in parameter_list]
    lt_3_3 = pd.DataFrame(lt_3_3)
    lt_3_3['servers'] = [parameters.num_servers
                         for parameters in parameter_list]

    # Setup the evaluators
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Setup the simulators
    heuristic_sim = Simulator(
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
        directory='./results/Heuristic/',
    )
    uncoded_sim = Simulator(
        solver=None,
        assignments=1,
        parameter_eval=analytic.uncoded_performance,
        directory='./results/Uncoded/',
    )

    # run simulations
    heuristic = heuristic_sim.simulate_parameter_list(parameter_list)
    uncoded = uncoded_sim.simulate_parameter_list(parameter_list)

    # include encoding/decoding delay
    heuristic.set_encode_delay(function=complexity.partitioned_encode_delay)
    heuristic.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    uncoded.set_uncoded(enable=True)

    settings_2_1 = {
        'label': 'LT $\epsilon_\mathsf{min}=0.2$, $P_\mathsf{f, target}=1e-1$',
        'color': 'g',
        'marker': 'x-',
        'linewidth': 2,
        'size': 7}
    settings_2_3 = {
        'label': 'LT $\epsilon_\mathsf{min}=0.2$, $P_\mathsf{f, target}=1e-3$',
        'color': 'b',
        'marker': 'd-',
        'linewidth': 2,
        'size': 7}
    settings_3_1 = {
        'label': 'LT $\epsilon_\mathsf{min}=0.3$, $P_\mathsf{f, target}=1e-1$',
        'color': 'k',
        'marker': 'x--',
        'linewidth': 2,
        'size': 7}
    settings_3_2 = {
        'label': 'LT $\epsilon_\mathsf{min}=0.3$, $P_\mathsf{f, target}=1e-3$',
        'color': 'c',
        'marker': 'd--',
        'linewidth': 2,
        'size': 8}
    settings_5 = {
        'label': 'LT tfp=1e-5',
        'color': 'c-',
        'marker': 'v',
        'linewidth': 3,
        'size': 8
    }
    settings_heuristic = {
        'label': 'BDC, Heuristic',
        'color': 'r',
        'marker': 'H-',
        'linewidth': 3,
        'size': 8
    }

    load_delay_plot(
        [heuristic,
         lt_2_1,
         lt_2_3,
         lt_3_1,
         lt_3_3],
        [settings_heuristic,
         settings_2_1,
         settings_2_3,
         settings_3_1,
         settings_3_2],
        'num_columns',
        xlabel='$\mathsf{Vectors}\;N$',
        normalize=uncoded,
        show=False,
    )
    plt.savefig('./plots/journal/lt.pdf')
    plt.show()
    return

def load_delay_plots():
    '''load/delay plots as function of partitions and size'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning_2()
    size_parameters = plot.get_parameters_size_2()[0:-4] # -2

    # Setup the simulators
    heuristic_sim = Simulator(
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
        directory='./results/Heuristic/',
    )

    random_sim = Simulator(
        solver=RandomSolver(),
        assignments=100,
        assignment_eval=sample_100,
        directory='./results/Random_100/',
    )

    hybrid_sim = Simulator(
        solver=AssignmentLoader(directory='./results/Hybrid/assignments/'),
        assignments=1,
        assignment_eval=sample_1000,
        assignment_type=CachedAssignment,
        directory='./results/Hybrid/'
    )

    uncoded_sim = Simulator(
        solver=None,
        assignments=1,
        parameter_eval=analytic.uncoded_performance,
        directory='./results/Uncoded/',
    )

    cmapred_sim = Simulator(
        solver=None,
        assignments=1,
        parameter_eval=analytic.cmapred_performance,
        directory='./results/Cmapred/',
    )

    stragglerc_sim = Simulator(
        solver=None,
        assignments=1,
        parameter_eval=analytic.stragglerc_performance,
        directory='./results/Stragglerc/',
    )

    rs_sim = Simulator(
        solver=None,
        assignments=1,
        parameter_eval=analytic.mds_performance,
        directory='./results/RS/',
    )

    # lt code simulations are handled using the rateless module. the simulation
    # framework differs from that for the BDC and analytic.
    lt_partitions = [rateless.evaluate(
        partition_parameters[0],
        target_overhead=1.3,
        target_failure_probability=1e-1,
    )] * len(partition_parameters)
    lt_partitions = pd.DataFrame(lt_partitions)
    lt_partitions['partitions'] = [parameters.num_partitions
                                   for parameters in partition_parameters]

    lt_size = rateless.evaluate_parameter_list(
        size_parameters,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    )
    lt_size['servers'] = [parameters.num_servers
                          for parameters in size_parameters]

    # Simulate partition parameters
    heuristic_partitions = heuristic_sim.simulate_parameter_list(partition_parameters)
    hybrid_partitions = hybrid_sim.simulate_parameter_list(partition_parameters)
    random_partitions = random_sim.simulate_parameter_list(partition_parameters)
    rs_partitions = rs_sim.simulate_parameter_list(partition_parameters)
    uncoded_partitions = uncoded_sim.simulate_parameter_list(partition_parameters)
    cmapred_partitions = cmapred_sim.simulate_parameter_list(partition_parameters)
    stragglerc_partitions = stragglerc_sim.simulate_parameter_list(partition_parameters)

    # include encoding delay
    heuristic_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    hybrid_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    random_partitions.set_encode_delay(function=complexity.partitioned_encode_delay)
    rs_partitions.set_encode_delay(
        function=partial(complexity.partitioned_encode_delay, partitions=1)
    )
    stragglerc_partitions.set_encode_delay(function=complexity.stragglerc_encode_delay)

    # Include the reduce delay
    heuristic_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    hybrid_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_partitions.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_partitions.set_reduce_delay(
        function=partial(complexity.partitioned_reduce_delay, partitions=1)
    )
    uncoded_partitions.set_uncoded(enable=True)
    cmapred_partitions.set_cmapred(enable=True)
    stragglerc_partitions.set_reduce_delay(function=complexity.stragglerc_reduce_delay)
    stragglerc_partitions.set_stragglerc(enable=True)

    # Simulate size parameters
    heuristic_size = heuristic_sim.simulate_parameter_list(size_parameters)
    random_size = random_sim.simulate_parameter_list(size_parameters)
    rs_size = rs_sim.simulate_parameter_list(size_parameters)
    uncoded_size = uncoded_sim.simulate_parameter_list(size_parameters)
    cmapred_size = cmapred_sim.simulate_parameter_list(size_parameters)
    stragglerc_size = stragglerc_sim.simulate_parameter_list(size_parameters)

    # include encoding delay
    heuristic_size.set_encode_delay(function=complexity.partitioned_encode_delay)
    random_size.set_encode_delay(function=complexity.partitioned_encode_delay)
    rs_size.set_encode_delay(
        function=partial(complexity.partitioned_encode_delay, partitions=1)
    )
    stragglerc_size.set_encode_delay(function=complexity.stragglerc_encode_delay)

    # Include the reduce delay
    heuristic_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    random_size.set_reduce_delay(function=complexity.partitioned_reduce_delay)
    rs_size.set_reduce_delay(
        function=partial(complexity.partitioned_reduce_delay, partitions=1)
    )
    uncoded_size.set_uncoded(enable=True)
    cmapred_size.set_cmapred(enable=True)
    stragglerc_size.set_reduce_delay(function=complexity.stragglerc_reduce_delay)
    stragglerc_size.set_stragglerc(enable=True)

    # plot settings
    heuristic_plot_settings = {
        'label': r'BDC, Heuristic',
        'color': 'r',
        'marker': 'H-',
        'linewidth': 2,
        'size': 7}
    random_plot_settings = {
        'label': r'BDC, Random',
        'color': 'b',
        'marker': '^-',
        'linewidth': 2,
        'size': 8}
    hybrid_plot_settings = {
        'label': r'BDC, Hybrid',
        'color': 'c-',
        'marker': 'd',
        'linewidth': 2,
        'size': 6}
    lt_plot_settings = {
        'label': r'LT',
        'color': 'c',
        'marker': 'v',
        'linewidth': 3,
        'size': 8
    }
    cmapred_plot_settings = {
        'label': r'CMR',
        'color': 'g',
        'marker': 's--',
        'linewidth': 2,
        'size': 7}
    stragglerc_plot_settings = {
        'label': r'SC',
        'color': 'k',
        'marker': 'H',
        'linewidth': 2,
        'size': 7}
    rs_plot_settings = {
        'label': r'Unified',
        'color': 'k',
        'marker': 'd--',
        'linewidth': 2,
        'size': 7}

    # load/delay as function of num_partitions
    load_delay_plot(
        [heuristic_partitions,
         lt_partitions,
         cmapred_partitions,
         stragglerc_partitions,
         rs_partitions],
        [heuristic_plot_settings,
         lt_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'partitions',
        xlabel=r'$T$',
        normalize=uncoded_partitions,
        show=False,
    )
    plt.savefig('./plots/journal/partitions.pdf')

    # load/delay as function of system size
    load_delay_plot(
        [heuristic_size,
         lt_size,
         cmapred_size,
         stragglerc_size,
         rs_size],
        [heuristic_plot_settings,
         lt_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'servers',
        xlabel=r'$K$',
        normalize=uncoded_size,
        legend='load',
        show=False,
    )
    plt.savefig('./plots/journal/size.pdf')

    # load/delay for different solvers as function of num_partitions
    load_delay_plot(
        [heuristic_partitions,
         hybrid_partitions,
         random_partitions],
        [heuristic_plot_settings,
         hybrid_plot_settings,
         random_plot_settings],
        'partitions',
        xlabel=r'$T$',
        normalize=uncoded_partitions,
        show=False,
    )
    plt.savefig('./plots/journal/solvers_partitions.pdf')

    # load/delay as function of system size
    load_delay_plot(
        [heuristic_size,
         random_size],
        [heuristic_plot_settings,
         random_plot_settings],
        'servers',
        xlabel=r'$K$',
        normalize=uncoded_size,
        legend='load',
        show=False,
    )
    plt.savefig('./plots/journal/solvers_size.pdf')

    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # N_n_ratio_plots()
    lt_plots()
    # load_delay_plots()
