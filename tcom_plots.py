import math
import logging
import model
import plot
import rateless
import complexity
import simulation
import matplotlib.pyplot as plt
import pyrateless

from functools import partial
from scipy.special import comb as nchoosek
from evaluation import analytic
from evaluation.binsearch import SampleEvaluator
from solvers.heuristicsolver import HeuristicSolver
from solvers.randomsolver import RandomSolver
from solvers.assignmentloader import AssignmentLoader
from assignments.cached import CachedAssignment

# pyplot setup
plt.style.use('seaborn-paper')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['figure.dpi'] = 200

# import plot settings
# import evaluator functions
# import parameters
# figure out which LT code parameters we need to simulate
# run LT codes, R10 codes, and RQ codes overnight
# create one function for each plot


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

# Setup the evaluators
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
lt_fun = partial(
    simulation.simulate,
    directory='./results/LT_3_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.3,
        target_failure_probability=1e-1,
        cachedir='./results/LT_3_1/',
    ),
    rerun=True,
)

def get_parameters_partitioning():
    '''Constant system size, increasing partitions, num_outputs=num_columns'''
    rows_per_batch = 250
    num_servers = 9
    q = 6
    num_outputs = 6000
    server_storage = 1/3
    num_partitions = [
        2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30,
        40, 50, 60, 75, 100, 120, 125, 150, 200, 250,
        300, 375, 500, 600, 750, 1000, 1500, 3000,
    ]
    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=partitions,
        )
        parameters.append(par)

    return parameters

def get_parameters_size():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 1000
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor,
        )
        parameters.append(par)
    return parameters

def get_parameters_size_partitions():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 1000
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor,
        )
        for T in range(par.rows_per_batch, par.num_source_rows+1):
            try:
                dct = par.asdict()
                dct['num_partitions'] = T
                p = model.SystemParameters.fromdct(dct)
            except ValueError as err:
                continue
            parameters.append(p)
    return parameters

def get_parameters_workload(num_servers, W=1e8, num_partitions=None,
                            code_rate=2/3, muq=2, tol=0.05):
    '''Get a list of parameters for the size plot.'''
    q = code_rate*num_servers
    server_storage = muq / q

    # assume num_outputs and num_columns is a constant factor of
    # num_source_rows
    alpha = 0.01
    root = (W/(alpha**2*server_storage)) ** (1. / 3)
    num_source_rows = round(root)
    num_outputs = round(alpha*num_source_rows / q) * q
    num_columns = num_outputs

    rows_per_server = num_source_rows * server_storage
    num_batches = nchoosek(num_servers, muq, exact=True)
    num_coded_rows = num_source_rows / code_rate
    rows_per_batch = round(num_coded_rows / num_batches)
    if not num_partitions:
        num_partitions = rows_per_batch
    # print(num_servers, num_coded_rows, num_batches, num_coded_rows/num_batches)
    if num_coded_rows / num_batches < 1:
        raise ValueError()

    m = model.SystemParameters(
        rows_per_batch=rows_per_batch,
        num_servers=num_servers,
        q=q,
        num_outputs=num_outputs,
        server_storage=server_storage,
        num_partitions=num_partitions,
        num_columns=num_columns,
    )
    W_emp = workload(m)
    err = abs((W-W_emp)/W)
    if err > tol:
        raise ValueError("err={} too big".format(err))
    return m

def get_parameters_constant_workload(all_T=False):
    l = list()
    W_target = 1e8
    min_source_rows = 0 # ensure number of source rows is always increasing
    # for i in range(6, 301):
    for i in range(6, 200):
        try:
            m = get_parameters_workload(i, W=W_target)
        except ValueError as err:
            continue
        if m.num_source_rows <= min_source_rows:
            continue
        min_source_rows = m.num_source_rows
        l.append(m)
        if all_T:
            for T in range(m.rows_per_batch+1, m.num_source_rows):
                try:
                    m = get_parameters_workload(i, W=W_target, num_partitions=T)
                except ValueError as err:
                    continue
                l.append(m)

    for m in l:
        W = workload(m)
        print(m)
        print(W/W_target)
        print()
    return l

def workload(p):
    return p.server_storage*p.num_source_rows*p.num_columns*p.num_outputs

def lt_parameters():
    # parameters = plot.get_parameters_size()[:-2] # -2
    parameters = plot.get_parameters_partitioning()
    for p in parameters:
        m = p.num_source_rows
        c, delta = pyrateless.heuristic(
            num_inputs=m,
            target_failure_probability=1e-1,
            target_overhead=1.3,
        )
        mode = pyrateless.coding.stats.mode_from_delta_c(
            num_inputs=m,
            delta=delta,
            c=c,
        )
        print('m={}, delta={}, mode={}'.format(
            m, delta, mode,
        ))

        m = int(p.num_source_rows / p.rows_per_batch)
        c, delta = pyrateless.heuristic(
            num_inputs=m,
            target_failure_probability=1e-1,
            target_overhead=1.3,
        )
        mode = pyrateless.coding.stats.mode_from_delta_c(
            num_inputs=m,
            delta=delta,
            c=c,
        )
        print('m={}, delta={}, mode={}'.format(
            m, delta, mode,
        ))
    return

def partition_plot():
    parameters = get_parameters_partitioning()

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    rs = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='fft',
        ),
    )
    hybrid = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=hybrid_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )
    random = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=random_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )
    cmapred = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=cmapred_fun,
        map_complexity_fun=complexity.map_complexity_cmapred,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    stragglerc = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
    )
    # lt = simulation.simulate_parameter_list(
    #     parameter_list=parameters,
    #     simulate_fun=lt_fun,
    #     map_complexity_fun=complexity.map_complexity_unified,
    #     encode_delay_fun=False,
    #     reduce_delay_fun=False,
    # )

    # load/delay as function of system size
    plot.load_delay_plot(
        [heuristic,
         # lt,
         cmapred,
         stragglerc,
         rs],
        [heuristic_plot_settings,
         # lt_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        vline=parameters[0].rows_per_batch,
        ylim_top=(0.5, 1.1),
        ylim_bot=(0.5,4),
        xlim_bot=(2, 3000),
    )
    plt.savefig('./plots/tcom/partitions.png', dpi='figure')

    plot.load_delay_plot(
        [heuristic,
         hybrid,
         random],
        [heuristic_plot_settings,
         hybrid_plot_settings,
         random_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        vline=parameters[0].rows_per_batch,
        ylim_top=(0.5, 0.6),
        ylim_bot=(1.5, 4),
        xlim_bot=(2, 3000),
    )
    plt.savefig('./plots/tcom/partitions_solvers.png', dpi='figure')

    plot.encode_decode_plot(
        [heuristic,
         # lt,
         cmapred,
         stragglerc,
         rs],
        [heuristic_plot_settings,
         # lt_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        legend='load',
        ncol=2,
        show=False,
        # xlim_bot=(6, 201),
    )

    plt.show()
    return

def size_plot():
    parameters = get_parameters_size()

    # same as above but with all possible partitioning levels
    parameters_all_t = get_parameters_size_partitions()

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    rs = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='fft',
        ),
    )
    rs_all_t = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='fft',
        ),
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )

    # filter out rows with low more than 1% over that of the RS code
    heuristic = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.01, :
    ]

    # find the optimal partitioning level for each number of servers
    heuristic = heuristic.loc[
        heuristic.groupby("num_servers")["overall_delay"].idxmin(), :
    ]
    heuristic.reset_index(inplace=True)

    random = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=random_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )
    cmapred = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=cmapred_fun,
        map_complexity_fun=complexity.map_complexity_cmapred,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    stragglerc = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=stragglerc_fun,
        map_complexity_fun=complexity.map_complexity_stragglerc,
        encode_delay_fun=complexity.stragglerc_encode_delay,
        reduce_delay_fun=complexity.stragglerc_reduce_delay,
    )

    plot.load_delay_plot(
        [rs,
         heuristic,
         cmapred,
         stragglerc],
        [rs_plot_settings,
         heuristic_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0.4, 1.1),
        ylim_bot=(0.25, 2.25),
    )
    plt.savefig('./plots/tcom/size.png', dpi='figure')

    plot.load_delay_plot(
        [heuristic,
         random],
        [heuristic_plot_settings,
         random_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0.4, 0.7),
        ylim_bot=(0.8, 2),
    )
    plt.savefig('./plots/tcom/size_solvers.png', dpi='figure')

    plot.encode_decode_plot(
        [rs,
         heuristic,
         random,
         cmapred,
         stragglerc],
        [rs_plot_settings,
         heuristic_plot_settings,
         random_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 201),
    )

    plt.show()
    return

def workload_plot():
    parameters = get_parameters_constant_workload()
    parameters_all_t = get_parameters_constant_workload(all_T=True)

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    rs = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='fft',
        ),
    )
    rs_all_t = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        simulate_fun=rs_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            partitions=1,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='fft',
        ),
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters_all_t,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='gen',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )

    # filter out rows with load more than 1% over that of the RS code
    heuristic_100 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.00000000000000001, :
    ]
    heuristic_101 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.01, :
    ]
    heuristic_110 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.10, :
    ]

    # find the optimal partitioning level for each number of servers
    heuristic_100 = heuristic_100.loc[
        heuristic_100.groupby("num_servers")["overall_delay"].idxmin(), :
    ]
    heuristic_100.reset_index(inplace=True)
    heuristic_101 = heuristic_101.loc[
        heuristic_101.groupby("num_servers")["overall_delay"].idxmin(), :
    ]
    heuristic_101.reset_index(inplace=True)
    heuristic_110 = heuristic_110.loc[
        heuristic_110.groupby("num_servers")["overall_delay"].idxmin(), :
    ]
    heuristic_110.reset_index(inplace=True)

    plot.load_delay_plot(
        [heuristic_100,
         heuristic_101,
         heuristic_110,
         rs],
        [heuristic_plot_settings,
         heuristic_plot_settings,
         heuristic_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 300),
        ylim_top=(0.4, 1),
        ylim_bot=(0, 60),
    )
    # plt.savefig('./plots/tcom/size.png', dpi='figure')
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # lt_parameters()
    # partition_plot()
    # size_plot()
    workload_plot()
