import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import plot
import simulation
import rateless
import ita_plots as ita
import complexity
import overhead
import model

from functools import partial

heuristic_plot_settings = {
    'label': r'BDC, Heuristic',
    'color': 'r',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
lt_plot_settings = {
    'label': r'LT',
    'color': 'c',
    'marker': 'v-',
    'linewidth': 4,
    'size': 2}
lt_partitioned_plot_settings = {
    'label': r'LT, Partitioned',
    'color': 'k',
    'marker': '^-',
    'linewidth': 3,
    'size': 2}
lt_nodec_plot_settings = {
    'label': r'LT, no decoding',
    'color': 'b',
    'marker': '^--',
    'linewidth': 4,
    'size': 2}
uncoded_plot_settings = {
    'label': r'Uncoded',
    'color': 'g',
    'marker': 'd-',
    'linewidth': 4,
    'size': 2}
rs_plot_settings = {
    'label': r'RS',
    'color': 'k',
    'marker': 'o--',
    'linewidth': 4,
    'size': 2}

def get_parameters_size():
    '''tune size plot parameters to find a case where the LT code has an advantage.'''
    rows_per_server = 2000
    rows_per_partition = 20 # optimal number of partitions in this case
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 10
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def get_parameters_partitioning():
    '''Get a list of parameters for the partitioning plot.'''
    rows_per_batch = 10
    num_servers = 201
    q = 134
    num_outputs_factor = 10
    num_outputs = num_outputs_factor*q
    server_storage = 2/q
    num_partitions = [3350, 6700, 8375, 13400, 16750]
    # num_partitions = [16750]
    parameters = list()
    for T in num_partitions:
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=T,
        )
        parameters.append(par)
    return parameters

def get_parameters_tradeoff():
    '''Get a list of parameters for the tradeoff plot.'''
    rows_per_batch = 10
    num_servers = 201
    q = 134
    num_outputs_factor = 10
    # for q 
    num_outputs = num_outputs_factor*q
    server_storage = 2/q
    num_partitions = [3350, 6700, 8375, 13400, 16750]
    # num_partitions = [16750]
    parameters = list()
    for T in num_partitions:
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=T,
        )
        parameters.append(par)
    return parameters

def size_plot():
    parameters = get_parameters_size()[:-2]
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=ita.lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_nodec = lt.copy()
    lt_nodec['overall_delay'] -= lt_nodec['reduce']
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=ita.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    plot.load_delay_plot(
        [heuristic,
         lt,
         lt_nodec],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_nodec_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded,
        show=False,
        legend='delay',
        xlim_bot=(20, 201),
        xlim_top=(20, 201),
        ylim_bot=(0.8, 2.0),
        ylim_top=(0.4, 1),
    )
    plt.savefig("./plots/180225/size.png")
    return

def partition_plot():
    parameters = get_parameters_partitioning()
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    # lt = simulation.simulate_parameter_list(
    #     parameter_list=parameters,
    #     simulate_fun=ita.lt_fun,
    #     map_complexity_fun=complexity.map_complexity_unified,
    #     encode_delay_fun=False,
    #     reduce_delay_fun=False,
    # )
    # lt_nodec = lt.copy()
    # lt_nodec['overall_delay'] -= lt_nodec['reduce']
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=ita.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    plot.load_delay_plot(
        [heuristic,
         # lt,
         # lt_nodec
        ],
        [heuristic_plot_settings,
         # lt_plot_settings,
         # lt_nodec_plot_settings
        ],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded,
        show=False,
        legend='delay',
    )
    return

def deadline_plot():
    parameters = plot.get_parameters_size()[-3]
    df = ita.heuristic_fun(parameters)
    samples_bdc = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=complexity.block_diagonal_encoding_complexity,
        reduce_complexity_fun=complexity.partitioned_reduce_complexity,
    )
    cdf_bdc = simulation.cdf_from_samples(samples_bdc)
    m = complexity.map_complexity_unified(parameters)
    print("encode", complexity.block_diagonal_encoding_complexity(parameters) / m)
    print("reduce", complexity.partitioned_reduce_complexity(parameters) / m)

    df = ita.lt_fun(parameters)
    order_values, order_probabilities = rateless.order_pdf(
        parameters=parameters,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    )
    print(order_values)
    print(order_probabilities)
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
    # m = complexity.map_complexity_unified(parameters)
    print("encode", df['encoding_multiplications'].mean() / parameters.num_servers / m)
    print("reduce", df['decoding_multiplications'].mean() / parameters.q / m)

    samples_lt_nodec = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=lambda x: df['encoding_multiplications'].mean(),
        reduce_complexity_fun=False,
        order_values=order_values,
        order_probabilities=order_probabilities,
    )
    cdf_lt_nodec = simulation.cdf_from_samples(
        samples_lt_nodec,
    )

    df = ita.uncoded_fun(parameters)
    samples_uncoded = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_complexity_fun=False,
        reduce_complexity_fun=False,
    )
    cdf_uncoded = simulation.cdf_from_samples(samples_uncoded)

    df = ita.rs_fun(parameters)
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

    # t = np.linspace(samples_lt.min(), samples_bdc.max())
    t = np.linspace(200, 800, 100)
    plt.figure()
    plt.hist(
        samples_lt,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_plot_settings['color'],
        label='LT',
    )
    plt.plot(t, cdf_lt(t), lt_plot_settings['color'])

    plt.hist(
        samples_bdc,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=heuristic_plot_settings['color'],
        label='BDC',
    )
    plt.plot(t, cdf_bdc(t), heuristic_plot_settings['color'])

    plt.hist(
        samples_uncoded,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=uncoded_plot_settings['color'],
        label='Uncoded',
    )
    plt.plot(t, cdf_uncoded(t), uncoded_plot_settings['color'])

    plt.rc('pgf',  texsystem='pdflatex')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    _ = plt.figure(figsize=(8,5))
    plt.autoscale(enable=True)
    ax1 = plt.gca()
    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.semilogy(
        t, 1-cdf_bdc(t),
        heuristic_plot_settings['color'],
        linewidth=2,
        label=r'BDC, Heuristic',
    )
    plt.semilogy(
        t, 1-cdf_uncoded(t),
        uncoded_plot_settings['color']+':',
        linewidth=4,
        label='Uncoded',
    )
    plt.semilogy(
        t, 1-cdf_lt(t),
        lt_plot_settings['color']+':',
        linewidth=4,
        label='LT',
    )
    # plt.semilogy(
    #     t, 1-cdf_lt_nodec(t),
    #     lt_nodec_plot_settings['color']+':',
    #     linewidth=4,
    #     label='LT, no decoding',
    # )
    plt.semilogy(
        t, 1-cdf_rs(t),
        rs_plot_settings['color']+'-',
        linewidth=4,
        label='Unified',
    )

    plt.ylabel(r'$\Pr(\rm{Delay} > t)$', fontsize=28)
    plt.xlabel(r'$t$', fontsize=28)

    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        loc='lower left',
        fancybox=False,
        borderaxespad=0.1,
    )

    plt.grid()
    plt.tight_layout()
    plt.xlim(t.min(), t.max())
    plt.ylim(1e-15, 1)
    plt.savefig("./plots/180225/deadline.png")
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    size_plot()
    # partition_plot()
    deadline_plot()
    plt.show()
