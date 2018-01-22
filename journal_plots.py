'''Script to create the plots for the journal paper

'''

import logging
import complexity
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import stats
import rateless
import plot
import simulation

from functools import partial
from plot import get_parameters_size, load_delay_plot
from plot import get_parameters_partitioning, get_parameters_partitioning_2
from plot import get_parameters_N
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

# plot settings
heuristic_plot_settings = {
    'label': r'BDC, Heuristic',
    'color': 'r',
    'marker': 'o-',
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
    'color': 'g',
    'marker': 's-',
    'linewidth': 2,
    'size': 6}
lt_plot_settings = {
    'label': r'LT',
    'color': 'b',
    'marker': 'v:',
    'linewidth': 3,
    'size': 8
}
lt_partitioned_plot_settings = {
    'label': r'LT, Partitioned',
    'color': 'c',
    'marker': '^-',
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
    'color': 'm',
    'marker': '*-',
    'linewidth': 2,
    'size': 7}
rs_plot_settings = {
    'label': r'Unified',
    'color': 'k',
    'marker': 'd--',
    'linewidth': 2,
    'size': 7}
uncoded_plot_settings = {
    'label': r'UC',
    'color': 'g',
    'marker': 'd-',
    'linewidth': 2,
    'size': 7}

def deadline_plots():
    '''deadline plots'''

    # get system parameters
    parameters = plot.get_parameters_size_2()[-3] # -3
    # parameters = plot.get_parameters_size_2()[0]

    # setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # setup the partial functions that handles running the simulations
    heuristic_fun = partial(
        simulation.simulate,
        directory='./results/Heuristic/',
        samples=1,
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
    )

    uncoded_fun = partial(
        simulation.simulate,
        directory='./results/Uncoded/',
        samples=1,
        parameter_eval=analytic.uncoded_performance,
    )

    cmapred_fun = partial(
        simulation.simulate,
        directory='./results/Cmapred/',
        samples=1,
        parameter_eval=analytic.cmapred_performance,
    )

    stragglerc_fun = partial(
        simulation.simulate,
        directory='./results/Stragglerc/',
        samples=1,
        parameter_eval=analytic.stragglerc_performance,
    )

    rs_fun = partial(
        simulation.simulate,
        directory='./results/RS/',
        samples=1,
        parameter_eval=analytic.mds_performance,
    )

    lt_fun = partial(
        simulation.simulate,
        directory='./results/LT_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
        ),
    )

    lt_fun_partitioned = partial(
        simulation.simulate,
        directory='./results/LT_partitioned_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
            partitioned=True,
        ),
    )

    df = heuristic_fun(parameters)
    samples_bdc = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=complexity.block_diagonal_encoding_complexity,
        reduce_complexity_fun=complexity.partitioned_reduce_complexity,
    )
    cdf_bdc = simulation.cdf_from_samples(samples_bdc)

    df = rs_fun(parameters)
    samples_rs = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=partial(
            complexity.block_diagonal_encoding_complexity,
            partitions=1,
        ),
        reduce_complexity_fun=partial(
            complexity.partitioned_reduce_complexity,
            partitions=1,
        ),
    )
    cdf_rs = simulation.cdf_from_samples(samples_rs)

    df = lt_fun(parameters)
    order_values, order_probabilities = rateless.order_pdf(
        parameters=parameters,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    )

    samples_lt = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=lambda x: df['encoding_multiplications'].mean(),
        reduce_complexity_fun=lambda x: df['decoding_multiplications'].mean(),
        order_values=order_values,
        order_probabilities=order_probabilities,
    )
    cdf_lt = simulation.cdf_from_samples(samples_lt)

    # df = lt_fun_partitioned(parameters)
    # order_values, order_probabilities = rateless.order_pdf(
    #     parameters=parameters,
    #     target_overhead=1.3,
    #     target_failure_probability=1e-1,
    #     partitioned=True,
    # )

    # samples_lt_partitioned = simulation.delay_samples(
    #     df,
    #     parameters=parameters,
    #     map_complexity_fun=complexity.map_complexity_unified,
    #     encode_complexity_fun=lambda x: df['encoding_multiplications'].mean(),
    #     reduce_complexity_fun=lambda x: df['decoding_multiplications'].mean(),
    #     order_values=order_values,
    #     order_probabilities=order_probabilities,
    # )
    # cdf_lt_partitioned = simulation.cdf_from_samples(samples_lt_partitioned)


    df = uncoded_fun(parameters)
    samples_uncoded = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_complexity_fun=False,
        reduce_complexity_fun=False,
    )
    cdf_uncoded = simulation.cdf_from_samples(samples_uncoded)

    # find points to evaluate the cdf at
    # t = np.linspace(
    #     1.5 * complexity.map_complexity_unified(parameters),
    #     5 * complexity.map_complexity_unified(parameters),
    # )
    # t = np.linspace(samples_uncoded.min(), 12500)
    t = np.linspace(samples_uncoded.min(), samples_rs.max())
    t_norm = t # / parameters.num_columns #  / complexity.map_complexity_unified(parameters)

    # plot 1-cdf with a log y axis
    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    _ = plt.figure(figsize=(10,9))
    plt.autoscale(enable=True)
    ax1 = plt.gca()
    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.semilogy(
        t_norm, 1-cdf_bdc(t),
        heuristic_plot_settings['color'],
        linewidth=2,
        label=r'BDC, Heuristic',
    )
    plt.semilogy(
        t_norm, 1-cdf_lt(t),
        lt_plot_settings['color']+':',
        linewidth=4,
        label='LT',
    )
    # plt.semilogy(
    #     t_norm, 1-cdf_lt_partitioned(t),
    #     lt_partitioned_plot_settings['color']+':s',
    #     linewidth=4,
    #     label='LT, Partitioned',
    # )
    plt.semilogy(
        t_norm, 1-cdf_uncoded(t),
        uncoded_plot_settings['color']+'-.',
        linewidth=2,
        label='UC',
    )
    plt.semilogy(
        t_norm, 1-cdf_rs(t),
        rs_plot_settings['color']+'--',
        linewidth=2,
        label='Unified',
    )
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.ylabel(r'$\Pr(\rm{Delay} > t)$', fontsize=28)
    plt.xlabel(r'$t$', fontsize=28)
    plt.ylim(ymin=1e-15)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./plots/journal/deadline.pdf')
    # plt.savefig('./plots/meetings/deadline.png')

    # plot 1-cdf normalized
    # plt.figure()
    # normalize = 1-cdf_uncoded(t)
    # plt.semilogy(t, (1-cdf_bdc(t))/normalize, label='bdc')
    # plt.semilogy(t, (1-cdf_rs(t))/normalize, label='unified')
    # plt.legend()
    # plt.grid()

    # plot the empiric and fitted cdf's
    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    _ = plt.figure(figsize=(10,9))
    plt.autoscale(enable=True)
    ax1 = plt.gca()
    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.plot(t, cdf_bdc(t), heuristic_plot_settings['color'])
    plt.hist(
        samples_bdc, bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=heuristic_plot_settings['color'],
        label='bdc',
    )

    plt.plot(t, cdf_rs(t), rs_plot_settings['color'])
    plt.hist(
        samples_rs,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=rs_plot_settings['color'],
        label='unified',
    )

    plt.plot(t, cdf_uncoded(t), uncoded_plot_settings['color'])
    plt.hist(
        samples_uncoded,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=uncoded_plot_settings['color'],
        label='uncoded',
    )

    plt.plot(t, cdf_lt(t), lt_plot_settings['color'])
    # plt.plot(t, cdf_lt_2(t), 'c')
    plt.hist(
        samples_lt, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_plot_settings['color'], label='lt')
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.ylabel(r'PDF', fontsize=28)
    plt.xlabel(r'$t$', fontsize=28)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./plots/meetings/pdf_0.png')
    plt.show()
    return

def encode_decode_plots():
    '''load/delay plots as function of partitions and size'''

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    partition_parameters = get_parameters_partitioning_2()
    parameter_list = plot.get_parameters_size_2()[0:-4] # -2

    # Setup the simulators
    heuristic_sim = Simulator(
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
        directory='./results/Heuristic/',
    )

    # lt code simulations are handled using the rateless module. the simulation
    # framework differs from that for the BDC and analytic.
    # lt_partitions = [rateless.evaluate(
    #     partition_parameters[0],
    #     target_overhead=1.3,
    #     target_failure_probability=1e-1,
    # )] * len(partition_parameters)
    # lt_partitions = pd.DataFrame(lt_partitions)
    # lt_partitions['partitions'] = [parameters.num_partitions
    #                                for parameters in partition_parameters]

    # lt_size = rateless.evaluate_parameter_list(
    #     size_parameters,
    #     target_overhead=1.3,
    #     target_failure_probability=1e-1,
    # )
    # lt_size['servers'] = [parameters.num_servers
    #                       for parameters in size_parameters]

    lt_2_1 = rateless.evaluate_parameter_list(
        parameter_list,
        target_overhead=1.2,
        target_failure_probability=1e-1,
    )
    lt_2_1['servers'] = [
        parameters.num_servers for parameters in parameter_list
    ]

    lt_2_1 = rateless.evaluate_parameter_list(
        parameter_list,
        target_overhead=1.2,
        target_failure_probability=1e-1,
    )
    lt_2_1['servers'] = [
        parameters.num_servers for parameters in parameter_list
    ]

    lt_2_3 = rateless.evaluate_parameter_list(
        parameter_list,
        target_overhead=1.2,
        target_failure_probability=1e-3,
    )
    lt_2_3['servers'] = [
        parameters.num_servers for parameters in parameter_list
    ]

    lt_3_1 = rateless.evaluate_parameter_list(
        parameter_list,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    )
    lt_3_1['servers'] = [
        parameters.num_servers for parameters in parameter_list
    ]

    lt_3_3 = rateless.evaluate_parameter_list(
        parameter_list,
        target_overhead=1.3,
        target_failure_probability=1e-3,
    )
    lt_3_3['servers'] = [
        parameters.num_servers for parameters in parameter_list
    ]

    # Simulate BDC heuristic
    heuristic = heuristic_sim.simulate_parameter_list(parameter_list)

    # include encoding delay
    heuristic.set_encode_delay(function=complexity.partitioned_encode_delay)

    # Include the reduce delay
    heuristic.set_reduce_delay(function=complexity.partitioned_reduce_delay)

    # plot settings
    settings_2_1 = {
        'label': r'LT $(0.2, 10^{-1})$',
        'color': 'g',
        'marker': 'x-',
        'linewidth': 2,
        'size': 7}
    settings_2_3 = {
        'label': r'LT $(0.2, 10^{-3})$',
        'color': 'b',
        'marker': 'd-',
        'linewidth': 2,
        'size': 7}
    settings_3_1 = {
        'label': r'LT $(0.3, 10^{-1})$',
        'color': 'k',
        'marker': 'x--',
        'linewidth': 2,
        'size': 7}
    settings_3_2 = {
        'label': r'LT $(0.3, 10^{-3})$',
        'color': 'c',
        'marker': 'd--',
        'linewidth': 2,
        'size': 8}
    settings_heuristic = {
        'label': 'BDC, Heuristic',
        'color': 'r',
        'marker': 'H-',
        'linewidth': 3,
        'size': 8
    }

    encode_decode_plot(
        [lt_2_1,
         lt_2_3,
         lt_3_1,
         lt_3_3],
        [settings_2_1,
         settings_2_3,
         settings_3_1,
         settings_3_2],
        xlabel=r'$K$',
        normalize=heuristic,
        show=False,
    )
    plt.show()
    return

    # encoding/decoding complexity as function of num_partitions
    # plot.encode_decode_plot(
    #     [lt_partitions],
    #     [lt_plot_settings],
    #     'partitions',
    #     xlabel=r'$T$',
    #     normalize=heuristic_partitions,
    #     show=False,
    # )
    # plt.savefig('./plots/journal/complexity_partitions.pdf')

    # encoding/decoding complexity as function of system size
    plot.encode_decode_plot(
        [],
        [lt_plot_settings],
        'servers',
        xlabel=r'$K$',
        normalize=heuristic_size,
        legend='encode',
        show=False,
    )
    # plt.savefig('./plots/journal/complexity_size.pdf')
    plt.show()
    return

def lt_plots():

    # get system parameters
    # parameter_list = plot.get_parameters_size_2()[:-4]
    parameter_list = plot.get_parameters_N()

    rerun = True
    # lt_1_1_fun = partial(
    #     simulation.simulate,
    #     directory='./results/LT_1_1/',
    #     samples=1,
    #     parameter_eval=partial(
    #         rateless.evaluate,
    #         target_overhead=1.1,
    #         target_failure_probability=1e-1,
    #     ),
    #     rerun=rerun,
    # )
    lt_2_1_fun = partial(
        simulation.simulate,
        directory='./results/LT_2_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.2,
            target_failure_probability=1e-1,
        ),
        rerun=rerun,
    )
    lt_3_1_fun = partial(
        simulation.simulate,
        directory='./results/LT_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
        ),
        rerun=rerun,
    )
    lt_2_3_fun = partial(
        simulation.simulate,
        directory='./results/LT_2_3/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.2,
            target_failure_probability=1e-3,
        ),
        rerun=rerun,
    )
    lt_3_3_fun = partial(
        simulation.simulate,
        directory='./results/LT_3_3/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-3,
        ),
        rerun=rerun,
    )

    # lt_1_1 = simulation.simulate_parameter_list(
    #     parameter_list=parameter_list,
    #     simulate_fun=lt_1_1_fun,
    #     map_complexity_fun=complexity.map_complexity_unified,
    #     encode_delay_fun=False,
    #     reduce_delay_fun=False,
    # )
    lt_2_1 = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_2_1_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_3_1 = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_3_1_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_2_3 = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_2_3_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_3_3 = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_3_3_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    # Setup the evaluators
    sample_1000 = SampleEvaluator(num_samples=1000)

    heuristic_fun = partial(
        simulation.simulate,
        directory='./results/Heuristic/',
        samples=1,
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
    )

    uncoded_fun = partial(
        simulation.simulate,
        directory='./results/Uncoded/',
        samples=1,
        parameter_eval=analytic.uncoded_performance,
    )

    # run simulations
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

    # settings_1_1 = {
    #     'label': r'LT $(0.1, 10^{-1})$',
    #     'color': 'c',
    #     'marker': 'x:',
    #     'linewidth': 2,
    #     'size': 7}
    settings_2_1 = {
        'label': r'LT $(0.2, 10^{-1})$',
        'color': 'g',
        'marker': 's-',
        'linewidth': 4,
        'size': 8}
    settings_2_3 = {
        'label': r'LT $(0.2, 10^{-3})$',
        'color': 'k',
        'marker': '^--',
        'linewidth': 3,
        'size': 7}
    settings_3_1 = {
        'label': r'LT $(0.3, 10^{-1})$',
        'color': 'm',
        'marker': 'o-',
        'linewidth': 4,
        'size': 8}
    settings_3_3 = {
        'label': r'LT $(0.3, 10^{-3})$',
        'color': 'b',
        'marker': 'v--',
        'linewidth': 3,
        'size': 8}
    settings_heuristic = {
        'label': 'BDC, Heuristic',
        'color': 'r',
        'marker': 'H-',
        'linewidth': 3,
        'size': 8
    }

    load_delay_plot(
        [heuristic,
         # lt_1_1,
         lt_2_1,
         lt_2_3,
         lt_3_1,
         lt_3_3],
        [settings_heuristic,
         # settings_1_1,
         settings_2_1,
         settings_2_3,
         settings_3_1,
         settings_3_3],
        'num_columns',
        xlabel=r'$n$',
        normalize=uncoded,
        ncol=2,
        loc=(0.025, 0.125),
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
    size_parameters = plot.get_parameters_size_2()[0:-2] # -2

    # setup the partial functions that handles running the simulations
    heuristic_fun = partial(
        simulation.simulate,
        directory='./results/Heuristic/',
        samples=1,
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
    )

    random_fun = partial(
        simulation.simulate,
        directory='./results/Random_100/',
        samples=100,
        solver=RandomSolver(),
        assignment_eval=sample_100,
    )

    hybrid_fun = partial(
        simulation.simulate,
        directory='./results/Hybrid/',
        samples=1,
        solver=AssignmentLoader(directory='./results/Hybrid/assignments/'),
        assignment_type=CachedAssignment,
        assignment_eval=sample_1000,
    )

    uncoded_fun = partial(
        simulation.simulate,
        directory='./results/Uncoded/',
        samples=1,
        parameter_eval=analytic.uncoded_performance,
    )

    cmapred_fun = partial(
        simulation.simulate,
        directory='./results/Cmapred/',
        samples=1,
        parameter_eval=analytic.cmapred_performance,
    )

    stragglerc_fun = partial(
        simulation.simulate,
        directory='./results/Stragglerc/',
        samples=1,
        parameter_eval=analytic.stragglerc_performance,
    )

    rs_fun = partial(
        simulation.simulate,
        directory='./results/RS/',
        samples=1,
        parameter_eval=analytic.mds_performance,
    )

    rerun = False
    lt_fun = partial(
        simulation.simulate,
        directory='./results/LT_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
        ),
        rerun=rerun,
    )

    lt_fun_partitioned = partial(
        simulation.simulate,
        directory='./results/LT_partitioned_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
            partitioned=True,
        ),
        rerun=rerun,
    )

    # simulate partition parameters
    heuristic_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    hybrid_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=hybrid_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    random_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=random_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    rs_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
        ),
    )
    uncoded_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    cmapred_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=cmapred_fun,
        map_complexity_fun=complexity.map_complexity_cmapred,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    stragglerc_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
    )
    _ = simulation.simulate_parameter_list(
        parameter_list=[partition_parameters[0]],
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    _ = simulation.simulate_parameter_list(
        parameter_list=[partition_parameters[0]],
        simulate_fun=lt_fun_partitioned,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=lt_fun_partitioned,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    # simulate size parameters
    heuristic_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    random_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=random_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    rs_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
        ),
    )
    uncoded_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    cmapred_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=cmapred_fun,
        map_complexity_fun=complexity.map_complexity_cmapred,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    stragglerc_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
    )
    lt_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=lt_fun_partitioned,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    # encoding/decoding complexity as function of num_partitions
    plot.encode_decode_plot(
        [lt_partitions,
         lt_partitioned_partitions],
        [lt_plot_settings,
         lt_partitioned_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=heuristic_partitions,
        show=False,
    )
    plt.savefig('./plots/journal/complexity_partitions.pdf')

    # encoding/decoding complexity as function of system size
    plot.encode_decode_plot(
        [# lt_size,
         lt_partitioned_size],
        [# lt_plot_settings,
         lt_partitioned_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=heuristic_size,
        legend='load',
        show=False,
    )
    plt.savefig('./plots/journal/complexity_size.pdf')

    # load/delay as function of num_partitions
    plot.load_delay_plot(
        [heuristic_partitions,
         lt_partitions,
         lt_partitioned_partitions,
         cmapred_partitions,
         stragglerc_partitions,
         rs_partitions],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded_partitions,
        ncol=2,
        loc=(0.025, 0.125),
        vline=partition_parameters[0].rows_per_batch,
        show=False,
    )
    plt.savefig('./plots/journal/partitions.pdf')

    # load/delay as function of system size
    plot.load_delay_plot(
        [heuristic_size,
         lt_size,
         lt_partitioned_size,
         cmapred_size,
         stragglerc_size,
         rs_size],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded_size,
        legend='load',
        ncol=2,
        show=False,
    )
    plt.savefig('./plots/journal/size.pdf')

    # load/delay for different solvers as function of num_partitions
    plot.load_delay_plot(
        [heuristic_partitions,
         hybrid_partitions,
         random_partitions],
        [heuristic_plot_settings,
         hybrid_plot_settings,
         random_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded_partitions,
        vline=partition_parameters[0].rows_per_batch,
        show=False,
    )
    plt.savefig('./plots/journal/solvers_partitions.pdf')

    # load/delay as function of system size
    plot.load_delay_plot(
        [heuristic_size,
         random_size],
        [heuristic_plot_settings,
         random_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded_size,
        legend='load',
        show=False,
    )
    plt.savefig('./plots/journal/solvers_size.pdf')

    plt.show()
    return

def lt_partitioned_plot():

    # Setup the evaluators
    sample_100 = SampleEvaluator(num_samples=100)
    sample_1000 = SampleEvaluator(num_samples=1000)

    # Get parameters
    parameter_list = plot.get_parameters_size_2()[1:-4] # -2

    rerun = True

    # setup the partial functions that handles running the simulations
    heuristic_fun = partial(
        simulation.simulate,
        directory='./results/Heuristic/',
        samples=1,
        solver=HeuristicSolver(),
        assignment_eval=sample_1000,
    )
    lt_fun = partial(
        simulation.simulate,
        directory='./results/LT_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
        ),
        rerun=rerun,
    )
    lt_fun_partitioned = partial(
        simulation.simulate,
        directory='./results/LT_partitioned_3_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.3,
            target_failure_probability=1e-1,
            partitioned=True,
        ),
        rerun=rerun,
    )
    uncoded_fun = partial(
        simulation.simulate,
        directory='./results/Uncoded/',
        samples=1,
        parameter_eval=analytic.uncoded_performance,
    )

    # simulate parameters
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    lt = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=lt_fun_partitioned,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    uncoded_size = simulation.simulate_parameter_list(
        parameter_list=parameter_list,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

    # load/delay as function of system size
    plot.load_delay_plot(
        [heuristic,
         lt,
         lt_partitioned],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded_size,
        legend='load',
        ncol=2,
        show=False,
    )

    # plot encoding/decoding complexity
    plot.encode_decode_plot(
        [lt,
         lt_partitioned],
        [lt_plot_settings,
         lt_partitioned_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=heuristic,
        show=False,
    )
    plt.show()
    return

def lt_decoding_pdf_plot():

    parameters = plot.get_parameters_size_2()[0]
    target_overhead = 1.3
    target_failure_probability = 1e-1
    code_rate = parameters.q / parameters.num_servers
    # t = np.linspace(target_overhead, target_overhead + 0.0003)
    t = np.linspace(target_overhead, 1 / code_rate)

    plt.figure()

    # long code
    num_inputs = parameters.num_source_rows
    c, delta, mode = rateless.optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )
    decoding_pdf = rateless.decoding_success_pdf(
        t,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )
    print('Long mean overhead', (t * decoding_pdf).sum())
    plt.plot(t, decoding_pdf, label='Long')

    # long code
    num_inputs = int(parameters.num_source_rows / parameters.rows_per_batch)
    c, delta, mode = rateless.optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )
    decoding_pdf = rateless.decoding_success_pdf(
        t,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )
    print('Partitioned mean overhead', (t * decoding_pdf).sum())
    plt.plot(t, decoding_pdf, label='Partitioned')

    plt.grid()
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # deadline_plots()
    # encode_decode_plots()
    lt_plots()
    # load_delay_plots()
    # lt_partitioned_plot()
    # lt_decoding_pdf_plot()
