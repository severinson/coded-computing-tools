import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simulation
import complexity
import tcom_plots
import plot

from functools import partial
from matplotlib2tikz import save as tikz_save

# matplotlib2tikz requires the modification given in
# https://github.com/nschloe/matplotlib2tikz/issues/202 to display
# legend entries correctly

# pyplot setup
plt.style.use('seaborn-paper')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (4.2, 4.2)
plt.rcParams['figure.dpi'] = 300

parameters = tcom_plots.get_parameters_constant_workload()
parameters_all_t = tcom_plots.get_parameters_constant_workload(all_T=True)

def main():

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    # set tail scale
    map_complexity = 99220529
    # for tail_factor in [None, 0, 1/2, 1, 10, 100, 1000]:
    plot_params = [
        {'color': 'r', 'marker': 'o'},
        {'color': 'g', 'marker': 's'},
        {'color': 'b', 'marker': '^'},
        {'color': 'm', 'marker': 'd'},
    ]
    for tail_factor, pp in zip([0, 1, 10, 100], plot_params):
        plots(tail_factor, map_complexity, pp)

    plt.grid(True, which='both')
    plt.legend()
    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.xlim((6, 300))
    plt.ylim(1, pow(2, 5))
    tikz_save(
        './plots/180810/scale.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
    )
    # plt.savefig('./plots/180810/scale_plot.pdf', dpi='figure', bbox_inches='tight')
    plt.show()
    # tail_scale = None

    # foo = np.fromiter((complexity.map_complexity_unified(p) for p in parameters), dtype=float)
    # bar = (foo - foo.mean()) / foo
    # print(bar)
    # print(foo.mean())
    # return

# no tail, 1, 10, 100
def plots(tail_factor, map_complexity, plot_params):
    if tail_factor is None:
        tail_scale = None
    else:
        tail_scale = tail_factor * map_complexity

    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        tail_scale=tail_scale,
        simulate_fun=tcom_plots.uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        # reduce_delay_fun=lambda x: 0,
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            tail_scale=tail_scale,
            algorithm='uncoded',
        ),
    )
    rs = simulation.simulate_parameter_list(
        parameter_list=parameters,
        tail_scale=tail_scale,
        simulate_fun=tcom_plots.rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            tail_scale=tail_scale,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            tail_scale=tail_scale,
            algorithm='fft',
        ),
    )
    rs_all_t = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        tail_scale=tail_scale,
        simulate_fun=tcom_plots.rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            tail_scale=tail_scale,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            tail_scale=tail_scale,
            algorithm='fft',
        ),
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        tail_scale=tail_scale,
        simulate_fun=tcom_plots.heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            tail_scale=tail_scale,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            tail_scale=tail_scale,
            algorithm='bm',
        ),
    )

    # filter out rows with load more than 1% over that of the RS code
    # heuristic_100 = heuristic.loc[
    #     heuristic['load']/rs_all_t['load'] <= 1.00000000000000001, :
    # ]
    heuristic_101 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.01, :
    ]
    # heuristic_110 = heuristic.loc[
    #     heuristic['load']/rs_all_t['load'] <= 1.10, :
    # ]

    # find the optimal partitioning level for each number of servers
    # heuristic_100 = heuristic_100.loc[
    #     heuristic_100.groupby("num_servers")["overall_delay"].idxmin(), :
    # ]
    # heuristic_100.reset_index(inplace=True)
    heuristic_101 = heuristic_101.loc[
        heuristic_101.groupby("num_servers")["overall_delay"].idxmin(), :
    ]
    heuristic_101.reset_index(inplace=True)
    # heuristic_110 = heuristic_110.loc[
    #     heuristic_110.groupby("num_servers")["overall_delay"].idxmin(), :
    # ]
    # heuristic_110.reset_index(inplace=True)

    plt.loglog(
        heuristic_101['num_servers'],
        heuristic_101['overall_delay'] / uncoded['overall_delay'],
        plot_params['color']+'-'+plot_params['marker'],
        markerfacecolor='none', markeredgewidth=1.0,
        markevery=0.2,
        label=r'BDC, Heuristic, $\omega={}$'.format(tail_factor),
        basey=2,
    )
    plt.loglog(
        rs['num_servers'],
        rs['overall_delay'] / uncoded['overall_delay'],
        plot_params['color']+'--'+plot_params['marker'],
        markerfacecolor='none', markeredgewidth=1.0,
        markevery=0.2,
        label=r'Unified, $\omega={}$'.format(tail_factor),
        basey=2,
    )

    # plot.load_delay_plot(
    #     [heuristic_101,
    #      rs],
    #     [tcom_plots.heuristic_plot_settings,
    #      tcom_plots.rs_plot_settings],
    #     'num_servers',
    #     xlabel=r'$K$',
    #     normalize=uncoded,
    #     legend='load',
    #     ncol=2,
    #     show=False,
    #     title=r'$\beta={} \sigma_\mathsf{{map}}$'.format(tail_factor),
    #     xlim_bot=(6, 300),
    #     ylim_top=(0.4, 0.7),
    #     # ylim_bot=(0, 25),
    # )
    # plt.savefig('./plots/180810/scale_factor_{}.png'.format(tail_factor), dpi='figure', bbox_inches='tight')
    # plt.title(r'$\beta={} \sigma_\mathsf{{map}}$'.format(tail_factor))
    # plt.tight_layout()

if __name__ == '__main__':
    main()
