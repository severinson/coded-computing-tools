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

from functools import partial

logging.basicConfig(level=logging.DEBUG)

# Get parameters

size_parameters = plot.get_parameters_size_2()[0:-2]
size_parameters_3 = plot.get_parameters_size_3()[0:-2]
size_parameters_4 = plot.get_parameters_size_4()[0:-2]


partition_parameters = plot.get_parameters_partitioning_3()
deadline_parameters = plot.get_parameters_deadline()


heuristic_settings = {
    'label': r'BDC, $m/T=10$',
    'color': 'r',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
heuristic_3_settings = {
    'label': r'BDC, $m/T=100$',
    'color': 'k',
    'marker': 'v-',
    'linewidth': 4,
    'size': 2}
heuristic_4_settings = {
    'label': r'BDC, $T=\rm{limit}$',
    'color': 'g',
    'marker': '^:',
    'linewidth': 4,
    'size': 2}
lt_settings = {
    'label': r'LT',
    'color': 'c',
    'marker': 'o-',
    'linewidth': 4,
    'size': 2}
lt_nodec_settings = {
    'label': r'LT, no decoding',
    'color': 'b',
    'marker': '^--',
    'linewidth': 4,
    'size': 2}
uncoded_settings = {
    'label': r'Uncoded',
    'color': 'g',
    'marker': 'd-',
    'linewidth': 4,
    'size': 2}

lt_fun_2_1 = partial(
    simulation.simulate,
    directory='./results/LT_2_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.2,
        target_failure_probability=1e-1,
    ),
)

lt_fun_1_1 = partial(
    simulation.simulate,
    directory='./results/LT_1_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.1,
        target_failure_probability=1e-1,
    ),
)

lt_fun_3_1 = partial(
    simulation.simulate,
    directory='./results/LT_3_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.3,
        target_failure_probability=1e-1,
    ),
)

lt_fun_2_3 = partial(
    simulation.simulate,
    directory='./results/LT_2_3/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.2,
        target_failure_probability=1e-3,
    ),
)

def tradeoff_plot():
    heuristic_tradeoff = simulation.simulate_parameter_list(
        parameter_list=tradeoff_parameters,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    # heuristic_tradeoff['q'] = heuristic_tradeoff['muq'] / heuristic_tradeoff['server_storage']
    uncoded_tradeoff = simulation.simulate_parameter_list(
        parameter_list=tradeoff_parameters,
        simulate_fun=ita.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    plot.load_delay_plot(
        [heuristic_tradeoff,
        ],
        [heuristic_settings,
        ],
        'num_partitions',
        xlabel=r'$q$',
        normalize=uncoded_tradeoff,
        show=False,
    )
    plt.show()
    return

def partition_size_plots():

    heuristic_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    heuristic_partitions['q'] = heuristic_partitions['muq'] / heuristic_partitions['server_storage']

    heuristic_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    heuristic_3_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters_3,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    heuristic_4_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters_4,
        simulate_fun=ita.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )


    # lt_partitions = simulation.simulate_parameter_list(
    #     parameter_list=partition_parameters,
    #     simulate_fun=lt_fun_2_3,
    #     map_complexity_fun=complexity.map_complexity_unified,
    #     encode_delay_fun=False,
    #     reduce_delay_fun=False,
    # )
    # lt_partitions['q'] = lt_partitions['muq'] / lt_partitions['server_storage']
    # lt_partitions_nodec = lt_partitions.copy()
    # lt_partitions_nodec['overall_delay'] -= lt_partitions_nodec['reduce']

    lt_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=ita.lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_size_nodec = lt_size.copy()
    lt_size_nodec['overall_delay'] -= lt_size_nodec['reduce']

    uncoded_partitions = simulation.simulate_parameter_list(
        parameter_list=partition_parameters,
        simulate_fun=ita.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    uncoded_size = simulation.simulate_parameter_list(
        parameter_list=size_parameters,
        simulate_fun=ita.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

    plot.load_delay_plot(
        [heuristic_partitions,
         # lt_partitions,
         # lt_partitions_nodec
        ],
        [heuristic_settings,
         # lt_settings,
         # lt_nodec_settings
        ],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded_partitions,
        show=False,
    )

    plot.load_delay_plot(
        [heuristic_size,
         heuristic_3_size,
         heuristic_4_size,
         lt_size,
         lt_size_nodec],
        [heuristic_settings,
         heuristic_3_settings,
         heuristic_4_settings,
         lt_settings,
         lt_nodec_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded_size,
        show=False,
        legend='delay',
        xlim_bot=(6, 201),
        xlim_top=(6, 201),
        ylim_bot=(0.8, 1.8),
        ylim_top=(0.4, 1),
    )
    # plt.savefig("./plots/180223/size.png")
    return

def deadline_plot():

    # get system parameters
    # parameters = plot.get_parameters_size_3()[-3]
    # parameters = plot.get_parameters_N()[-1]
    parameters = plot.get_parameters_deadline()
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
    # df = lt_fun_2_1(parameters)
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
    m = complexity.map_complexity_unified(parameters)
    print("encode", df['encoding_multiplications'].mean() / m)
    print("reduce", df['decoding_multiplications'].mean() / m)

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

    t = np.linspace(samples_lt.min(), samples_bdc.max())

    plt.figure()
    plt.hist(
        samples_lt,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_settings['color'],
        label='LT',
    )
    plt.plot(t, cdf_lt(t), lt_settings['color'])

    plt.hist(
        samples_bdc,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=heuristic_settings['color'],
        label='BDC',
    )
    plt.plot(t, cdf_bdc(t), heuristic_settings['color'])

    plt.hist(
        samples_uncoded,
        bins=100,
        density=True,
        cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=uncoded_settings['color'],
        label='Uncoded',
    )
    plt.plot(t, cdf_uncoded(t), uncoded_settings['color'])

    # plt.figure()
    # plt.plot(t, cdf_lt(t))

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
        heuristic_settings['color'],
        linewidth=2,
        label=r'BDC, Heuristic',
    )
    plt.semilogy(
        t, 1-cdf_uncoded(t),
        uncoded_settings['color']+':',
        linewidth=4,
        label='Uncoded',
    )
    plt.semilogy(
        t, 1-cdf_lt(t),
        lt_settings['color']+':',
        linewidth=4,
        label='LT',
    )
    plt.semilogy(
        t, 1-cdf_lt_nodec(t),
        lt_nodec_settings['color']+':',
        linewidth=4,
        label='LT, no decoding',
    )

    plt.ylabel(r'$\Pr(\rm{Delay} > t)$', fontsize=28)
    plt.xlabel(r'$t$', fontsize=28)

    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        loc='upper right',
        fancybox=False,
        borderaxespad=0.1,
    )

    plt.grid()
    plt.tight_layout()
    plt.xlim(samples_bdc.min(), samples_lt.max())
    plt.ylim(1e-15, 1)
    # plt.savefig("./plots/180223/deadline_2.png")
    return

# p = plot.get_parameters_deadline()
# order = list(range(1, p.q))
# samples = 1000
# for o in [1.432, 1.433]:
#     servers = 0
#     for order in overhead.random_completion_orders(p, samples):
#         dct = overhead.delay_from_order(p, order, o)
#         servers += dct['servers']
#     servers /= samples
#     print(o, servers)


# tradeoff_plot()
partition_size_plots()
# deadline_plot()
plt.show()
