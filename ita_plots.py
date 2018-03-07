'''Script to generate plots for the ITA presentation.

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
from matplotlib2tikz import save as tikz_save
from plot import get_parameters_size, load_delay_plot
from plot import get_parameters_partitioning, get_parameters_partitioning_2
from plot import get_parameters_N
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.randomsolver import RandomSolver
from solvers.heuristicsolver import HeuristicSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

# Setup the evaluators
sample_100 = SampleEvaluator(num_samples=100)
sample_1000 = SampleEvaluator(num_samples=1000)

# Get parameters
partition_parameters = get_parameters_partitioning_2()
size_parameters = plot.get_parameters_size_2()[0:-2] # -2


tradeoff_parameters = plot.get_parameters_tradeoff()


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
    rerun=True,
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

# plot settings
heuristic_plot_settings = {
    'label': r'BDC, Heuristic',
    'color': 'r',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
random_plot_settings = {
    'label': r'BDC, Random',
    'color': 'b',
    'marker': '^-',
    'linewidth': 4,
    'size': 2}
hybrid_plot_settings = {
    'label': r'BDC, Hybrid',
    'color': 'c',
    'marker': 'd-',
    'linewidth': 2,
    'size': 2}
lt_plot_settings = {
    'label': r'LT',
    'color': 'c',
    'marker': 'v',
    'linewidth': 4,
    'size': 2}
lt_partitioned_plot_settings = {
    'label': r'LT, Partitioned',
    'color': 'c',
    'marker': '^-',
    'linewidth': 3,
    'size': 2}
cmapred_plot_settings = {
    'label': r'CMR',
    'color': 'g',
    'marker': 's--',
    'linewidth': 2,
    'size': 2}
stragglerc_plot_settings = {
    'label': r'SC',
    'color': 'k',
    'marker': 'H',
    'linewidth': 2,
    'size': 2}
rs_plot_settings = {
    'label': r'RS',
    'color': 'k',
    'marker': 'o--',
    'linewidth': 4,
    'size': 2}
rs_no_encdec_plot_settings = {
    'label': r'RS excl. encoding/decoding',
    'color': 'c',
    'marker': 'd-',
    'linewidth': 4,
    'size': 2}
uncoded_plot_settings = {
    'label': r'Uncoded',
    'color': 'g',
    'marker': 'd-',
    'linewidth': 4,
    'size': 2}

def deadline_plots():
    '''deadline plots'''

    # get system parameters
    parameters = plot.get_parameters_size_2()[-3]

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
    _ = plt.figure(figsize=(8,5))
    plt.autoscale(enable=True)
    ax1 = plt.gca()
    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.semilogy(
        t_norm, 1-cdf_uncoded(t),
        uncoded_plot_settings['color']+'-.',
        linewidth=2,
        label='Uncoded',
    )
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
        t_norm, 1-cdf_rs(t),
        rs_plot_settings['color']+'--',
        linewidth=2,
        label='RS',
    )
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        loc='lower right',
        fancybox=False,
        borderaxespad=0.1,
    )
    ax1.set_xticklabels([]) # remove x tick labels
    plt.ylabel(r'$\Pr(\rm{Delay} > t)$', fontsize=28)
    plt.xlabel(r'$t$', fontsize=28)
    plt.ylim(ymin=1e-15)
    plt.tight_layout()
    plt.grid()
    # plt.savefig('./plots/ita/deadline.pdf')
    tikz_save(
        './plots/ita/deadline.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )

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
    plt.show()
    return

def main():

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
    stragglerc_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
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
    rs_size_no_encdec = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
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
    stragglerc_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
    )

    rs_tradeoff_no_encdec = simulation.simulate_parameter_list(
        parameter_list=tradeoff_parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    uncoded_tradeoff = simulation.simulate_parameter_list(
        parameter_list=tradeoff_parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

    plot.load_delay_plot(
        [rs_size_no_encdec,
         rs_size],
        [rs_no_encdec_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        show=False,
        # loc='lower right',
        # ylim_bot=(0.3, 6.2),
    )
    # plt.savefig('./plots/ita/unified.pdf')
    tikz_save(
        './plots/ita/unified.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )

    # plt.rc('pgf',  texsystem='pdflatex')
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    # _ = plt.figure(figsize=(11,4))
    # plt.autoscale(enable=True)
    # plt.tight_layout()
    # ax1 = plt.subplot()
    # plt.setp(ax1.get_xticklabels(), fontsize=25)
    # plt.setp(ax1.get_yticklabels(), fontsize=25)
    # plot.plot_result(
    #     rs_size_no_encdec,
    #     rs_no_encdec_plot_settings,
    #     'num_servers',
    #     'overall_delay',
    #     ylabel=r'Delay',
    #     subplot=True,
    #     normalize=uncoded_size,
    # )
    # plot.plot_result(
    #     rs_size,
    #     rs_plot_settings,
    #     'num_servers',
    #     'overall_delay',
    #     ylabel=r'Delay',
    #     xlabel=r'Servers $K$',
    #     subplot=True,
    #     normalize=uncoded_size,
    # )
    # plt.margins(y=0.1)
    # plt.legend(
    #     numpoints=1,
    #     shadow=True,
    #     labelspacing=0,
    #     columnspacing=0.05,
    #     fontsize=22,
    #     loc='best',
    #     fancybox=False,
    #     borderaxespad=0.1,
    # )
    # plt.autoscale(enable=True)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0.12)
    # plt.margins(y=0.1)
    # tikz_save(
    #     './plots/ita/unified.tex',
    #     figurewidth='\\figurewidth',
    #     figureheight='\\figureheight',
    # )

    # plt.figure()
    # plt.plot(uncoded_tradeoff['overall_delay'])
    # plt.plot(rs_tradeoff_no_encdec['overall_delay'])
    # plt.figure()
    # plt.plot(uncoded_tradeoff['load'])
    # plt.plot(rs_tradeoff_no_encdec['load'])
    # plt.show()
    # return

    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    _ = plt.figure(figsize=(11,4))
    plt.autoscale(enable=True)
    plt.tight_layout()
    ax1 = plt.subplot()
    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.grid(True, which='both')
    plt.ylabel(r'Load', fontsize=28)
    plt.xlabel(r'Delay', fontsize=28)
    plt.autoscale()

    plot_settings = rs_no_encdec_plot_settings
    label = plot_settings['label']
    color = plot_settings['color']
    style = color + plot_settings['marker']
    linewidth = plot_settings['linewidth']
    size = plot_settings['size']
    xarray = rs_tradeoff_no_encdec['overall_delay']
    xarray /= uncoded_tradeoff['overall_delay']
    yarray = rs_tradeoff_no_encdec['load']
    yarray /= uncoded_tradeoff['load']
    plt.plot(
        xarray,
        yarray,
        style,
        label=label,
        linewidth=linewidth,
        markersize=size,
    )
    plt.margins(y=0.1)
    # plt.legend(
    #     numpoints=1,
    #     shadow=True,
    #     labelspacing=0,
    #     columnspacing=0.05,
    #     fontsize=22,
    #     loc='best',
    #     fancybox=False,
    #     borderaxespad=0.1,
    # )
    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.12)
    plt.margins(y=0.1)
    tikz_save(
        './plots/ita/unified_tradeoff.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )

    # load/delay as function of system size
    plot.load_delay_plot(
        [rs_size],
        [rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        legend='delay',
        ylim_top=(0.38, 1),
        ylim_bot=(0.5, 6),
        show=False,
    )
    plt.savefig('./plots/ita/size_1.pdf')
    # plt.savefig('./plots/ita/size_1.svg')

    plot.load_delay_plot(
        [random_size,
         rs_size],
        [random_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        legend='delay',
        ylim_top=(0.38, 1),
        ylim_bot=(0.5, 6),
        show=False,
    )
    plt.savefig('./plots/ita/size_2.pdf')
    # plt.savefig('./plots/ita/size_2.svg')

    plot.load_delay_plot(
        [heuristic_size,
         random_size,
         rs_size],
        [heuristic_plot_settings,
         random_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        legend='delay',
        ncol=2,
        ylim_top=(0.38, 1),
        ylim_bot=(0.5, 6),
        show=False,
    )
    plt.savefig('./plots/ita/size_3.pdf')
    # plt.savefig('./plots/ita/size_3.svg')

    plot.load_delay_plot(
        [heuristic_size,
         random_size,
         lt_size,
         rs_size,
         stragglerc_size],
        [heuristic_plot_settings,
         random_plot_settings,
         lt_plot_settings,
         rs_plot_settings,
         stragglerc_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        legend='delay',
        ncol=2,
        ylim_top=(0.38, 1),
        ylim_bot=(0.5, 6),
        show=False,
    )
    plt.savefig('./plots/ita/size_4.pdf')
    # plt.savefig('./plots/ita/size_4.svg')
    tikz_save(
        './plots/ita/size_1.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )

    plot.load_delay_plot(
        [heuristic_size,
         random_size,
         lt_size],
        [heuristic_plot_settings,
         random_plot_settings,
         lt_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        legend='delay',
        ylim_top=(0.38, 1),
        ylim_bot=(0.8, 2),
        show=False,
    )
    plt.savefig('./plots/ita/size_5.pdf')
    # plt.savefig('./plots/ita/size_5.svg')
    tikz_save(
        './plots/ita/size_2.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )

    # load/delay as function of num_partitions
    plot.load_delay_plot(
        [rs_partitions],
        [rs_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded_partitions,
        ncol=1,
        legend='load',
        vline=partition_parameters[0].rows_per_batch,
        show=False,
    )
    plt.savefig('./plots/ita/partitions_1.pdf')
    # plt.savefig('./plots/ita/partitions_1.svg')

    plot.load_delay_plot(
        [heuristic_partitions,
         random_partitions,
         rs_partitions,
         stragglerc_partitions],
        [heuristic_plot_settings,
         random_plot_settings,
         rs_plot_settings,
         stragglerc_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded_partitions,
        legend='load',
        vline=partition_parameters[0].rows_per_batch,
        show=False,
    )
    plt.savefig('./plots/ita/partitions_2.pdf')
    # plt.savefig('./plots/ita/partitions_2.svg')
    tikz_save(
        './plots/ita/partitions_1.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )
    plt.show()
    return

    plot.load_delay_plot(
        [heuristic_partitions,
         random_partitions,
         lt_partitions],
        [heuristic_plot_settings,
         random_plot_settings,
         lt_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded_partitions,
        ncol=1,
        legend='load',
        vline=partition_parameters[0].rows_per_batch,
        show=False,
    )
    plt.savefig('./plots/ita/partitions_3.pdf')
    # plt.savefig('./plots/ita/partitions_3.pgf')
    tikz_save(
        './plots/ita/partitions_2.tex',
        figurewidth='\\figurewidth',
        figureheight='\\figureheight',
    )
    # plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
    # deadline_plots()
