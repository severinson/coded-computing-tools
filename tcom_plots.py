import math
import logging
import numpy as np
import scipy.stats
import model
import plot
import rateless
import complexity
import simulation
import matplotlib.pyplot as plt
import pyrateless
import overhead

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
plt.rcParams['figure.figsize'] = (4.2, 4.2)
plt.rcParams['figure.dpi'] = 300

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
    'marker': 'v-',
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

# LT code simulations have to be re-run if n or N have been changed.
rerun = False
lt_fun = partial(
    simulation.simulate,
    directory='./results/LT_3_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.3,
        target_failure_probability=1e-1,
        cachedir='./results/LT_3_1/overhead',
    ),
    rerun=rerun,
)
lt_partitioned_fun = partial(
    simulation.simulate,
    directory='./results/LT_3_1_partitioned/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.3,
        target_failure_probability=1e-1,
        partitioned=True,
        cachedir='./results/LT_3_1_partitioned/overhead',
    ),
    rerun=rerun,
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
    num_outputs_factor = 10
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
        par.num_columns *= 100
        parameters.append(par)
    return parameters

def get_parameters_size_partitions():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 10
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
                dct['num_columns'] = 100*par.num_columns
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
    for i in range(6, 301):
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

def get_parameters_N():
    '''Get a list of parameters for the N to n ratio plot.'''
    rows_per_batch = 100
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/3
    num_partitions = 240
    num_outputs = 10*q
    parameters = list()
    for i in range(1, 11):
        # num_outputs = i * q
        num_columns = pow(i, 2) * 200
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=num_partitions,
            num_columns=num_columns,
        )
        parameters.append(par)

    return parameters

def lt_parameters(tfp=1e-1, to=1.3, partitioned=False):
    parameters = get_parameters_N()
    for p in parameters:
        if partitioned:
            num_inputs = int(p.num_source_rows / p.rows_per_batch)
        else:
            num_inputs = p.num_source_rows

        c, delta = pyrateless.heuristic(
            num_inputs=num_inputs,
            target_failure_probability=tfp,
            target_overhead=to,
        )
        mode = pyrateless.coding.stats.mode_from_delta_c(
            num_inputs=num_inputs,
            delta=delta,
            c=c,
        )
        print('num_inputs={}, delta={}, mode={}'.format(
            num_inputs, delta, mode,
        ))

    return

def lt_plots():
    parameters = get_parameters_N()
    [print(p) for p in parameters]
    print()
    lt_parameters(to=1.2, tfp=1e-1)
    print()
    lt_parameters(to=1.3, tfp=1e-1)
    print()
    lt_parameters(to=1.2, tfp=1e-3)
    print()
    lt_parameters(to=1.3, tfp=1e-3)
    return

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    rerun = True
    lt_2_1_fun = partial(
        simulation.simulate,
        directory='./results/LT_2_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.2,
            target_failure_probability=1e-1,
            cacheir='./results/LT_2_1/overhead',
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
            cacheir='./results/LT_3_1/overhead',
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
            cacheir='./results/LT_2_3/overhead',
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
            cacheir='./results/LT_3_3/overhead',
        ),
        rerun=rerun,
    )

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
        parameter_list=parameter,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

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
         lt_2_1,
         lt_2_3,
         lt_3_1,
         lt_3_3],
        [settings_heuristic,
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
    # plt.savefig('./plots/tcom/lt.pdf')
    plt.show()
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
    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_partitioned_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    # load/delay as function of system size
    plot.load_delay_plot(
        [heuristic,
         lt,
         lt_partitioned,
         cmapred,
         stragglerc,
         rs],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded,
        legend='delay',
        ncol=2,
        show=False,
        vline=parameters[0].rows_per_batch,
        ylim_top=(0.5, 1.1),
        ylim_bot=(0.5,4),
        xlim_bot=(2, 3000),
    )
    plt.savefig('./plots/tcom/partitions.png', dpi='figure')
    plt.savefig('./plots/tcom/partitions.pdf', dpi='figure', bbox_inches='tight')

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
    plt.savefig('./plots/tcom/solvers_partitions.png', dpi='figure')
    plt.savefig('./plots/tcom/solvers_partitions.pdf', dpi='figure', bbox_inches='tight')

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

    # plt.show()
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
    print(heuristic['overall_delay'])

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
    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    print(lt['overall_delay'])
    lt_partitioned = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_partitioned_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    print(lt_partitioned['overall_delay'])

    plot.load_delay_plot(
        [heuristic,
         lt,
         lt_partitioned,
         cmapred,
         stragglerc,
         rs],
        [heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         cmapred_plot_settings,
         stragglerc_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 201),
        # ylim_top=(0.4, 1.1),
        # ylim_bot=(0, 6),
    )
    # plt.savefig('./plots/tcom/size.png', dpi='figure')
    # plt.savefig('./plots/tcom/size.pdf', dpi='figure', bbox_inches='tight')

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
        # ylim_top=(0.4, 0.7),
        # ylim_bot=(0.5, 3),
    )
    # plt.savefig('./plots/tcom/solvers_size.png', dpi='figure')
    # plt.savefig('./plots/tcom/solvers_size.pdf', dpi='figure', bbox_inches='tight')

    plot.encode_decode_plot(
        [rs,
         heuristic,
         lt,
         lt_partitioned],
        [rs_plot_settings,
         heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        legend='load',
        ncol=2,
        show=False,
        normalize=uncoded,
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

    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_partitioned_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    plot.load_delay_plot(
        [# heuristic_100,
         heuristic_101,
         # heuristic_110,
         lt,
         lt_partitioned,
         rs],
        [# heuristic_plot_settings,
         heuristic_plot_settings,
         # heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 300),
        ylim_top=(0.4, 1),
        ylim_bot=(0, 25),
    )
    plt.savefig('./plots/tcom/workload.png', dpi='figure')
    plt.savefig('./plots/tcom/workload.pdf', dpi='figure', bbox_inches='tight')

    plot.encode_decode_plot(
        [# heuristic_100,
         heuristic_101,
         # heuristic_110,
         lt,
         lt_partitioned,
         rs],
        [# heuristic_plot_settings,
         heuristic_plot_settings,
         # heuristic_plot_settings,
         lt_plot_settings,
         lt_partitioned_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        legend='load',
        show=False,
        xlim_bot=(6, 300),
        # ylim_top=(0.4, 1),
        # ylim_bot=(0, 60),
    )

    # plt.show()
    return

def get_lt_cdf(parameters, partitioned=False):
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(partitioned, bool)
    if partitioned:
        filename = "lt_samples_partitioned"
        a=97.80714827099743
        loc=17005348.58022266
        scale=148050.2065582715
        return lambda t: scipy.stats.gamma.cdf(t, a, loc=loc, scale=scale)

        num_partitions = parameters.rows_per_batch
        df = lt_partitioned_fun(parameters)
        cachedir='./results/LT_3_1_partitioned/overhead'
        order_values, order_probabilities = rateless.order_pdf(
            parameters=parameters,
            target_overhead=1.3,
            target_failure_probability=1e-1,
            partitioned=partitioned,
        )
        # order_values = [
        #     134, 135, 136, 137, 138, 139, 140, 141, 142,
        #     143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
        #     157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        #     171, 172, 173, 174, 176, 177, 178, 179, 180, 182, 183, 185, 186, 188,
        #     191, 194, 200,
        # ]
        # order_probabilities = [
        #     9.95796511e-01, 2.20182752e-04, 3.00249207e-04, 2.00166138e-04,
        #     1.80149524e-04, 2.70224286e-04, 1.60132910e-04, 1.60132910e-04,
        #     1.60132910e-04, 1.40116297e-04, 2.10174445e-04, 1.30107990e-04,
        #     1.20099683e-04, 1.20099683e-04, 1.20099683e-04, 1.10091376e-04,
        #     5.00415345e-05, 1.00083069e-04, 1.00083069e-04, 1.00083069e-04,
        #     9.00747621e-05, 4.00332276e-05, 8.00664552e-05, 8.00664552e-05,
        #     4.00332276e-05, 8.00664552e-05, 4.00332276e-05, 7.00581483e-05,
        #     3.00249207e-05, 6.00498414e-05, 3.00249207e-05, 3.00249207e-05,
        #     6.00498414e-05, 3.00249207e-05, 3.00249207e-05, 3.00249207e-05,
        #     3.00249207e-05, 3.00249207e-05, 6.00498414e-05, 3.00249207e-05,
        #     2.00166138e-05, 2.00166138e-05, 2.00166138e-05, 2.00166138e-05,
        #     2.00166138e-05, 2.00166138e-05, 2.00166138e-05, 2.00166138e-05,
        #     2.00166138e-05, 2.00166138e-05, 2.00166138e-05, 2.00166138e-05,
        #     2.00166138e-05, 2.00166138e-05,
        # ]
    else:
        filename = "lt_samples"
        num_partitions = 1
        df = lt_fun(parameters)
        cachedir='./results/LT_3_1/overhead'
        order_values = [
            134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
            155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
            167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179,
            180, 182, 183, 185, 186, 188, 191, 194, 200,
        ]
        order_probabilities = [
            9.97919480e-01, 1.20030008e-04,
            1.80045011e-04, 1.00025006e-04, 1.00025006e-04, 1.50037509e-04,
            8.00200050e-05, 8.00200050e-05, 8.00200050e-05, 8.00200050e-05,
            1.10027507e-04, 6.00150038e-05, 6.00150038e-05, 6.00150038e-05,
            6.00150038e-05, 6.00150038e-05, 3.00075019e-05, 4.00100025e-05,
            4.00100025e-05, 4.00100025e-05, 4.00100025e-05, 2.00050013e-05,
            4.00100025e-05, 4.00100025e-05, 2.00050013e-05, 4.00100025e-05,
            2.00050013e-05, 4.00100025e-05, 1.00025006e-05, 2.00050013e-05,
            1.00025006e-05, 1.00025006e-05, 2.00050013e-05, 1.00025006e-05,
            1.00025006e-05, 1.00025006e-05, 1.00025006e-05, 1.00025006e-05,
            2.00050013e-05, 1.00025006e-05, 1.00025006e-05, 1.00025006e-05,
            1.00025006e-05, 1.00025006e-05, 1.00025006e-05, 1.00025006e-05,
            1.00025006e-05, 1.00025006e-05, 1.00025006e-05, 1.00025006e-05,
            1.00025006e-05, 1.00025006e-05, 1.00025006e-05, 1.00025006e-05,
        ]

    num_inputs = int(parameters.num_source_rows / num_partitions)
    encoding_complexity = rateless.lt_encoding_complexity(
        num_inputs=num_inputs,
        failure_prob=1e-1,
        target_overhead=1.3,
        code_rate=parameters.q/parameters.num_servers,
    )
    encoding_complexity *= parameters.num_columns
    encoding_complexity *= num_partitions
    encoding_complexity *= parameters.muq

    decoding_complexity = rateless.lt_decoding_complexity(
        num_inputs=num_inputs,
        failure_prob=1e-1,
        target_overhead=1.3,
    )
    decoding_complexity *= num_partitions
    decoding_complexity *= parameters.num_outputs

    samples = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=lambda x: encoding_complexity,
        reduce_complexity_fun=lambda x: decoding_complexity,
        order_values=order_values,
        order_probabilities=order_probabilities,
    )
    np.save(filename, samples)
    print("mean delay is", samples.mean())

    # if partitioned:
    #     # a=98.82425513382844
    #     # loc=166048.79856447875
    #     # scale=1509.6230105368877
    #     a=38.790654429629505
    #     loc=221431.70313319447
    #     scale=2419.151198191416
    #     return lambda t: scipy.stats.gamma.cdf(t, a, loc=loc, scale=scale), samples

    return simulation.cdf_from_samples(samples, n=100), samples

def deadline_plot():
    '''deadline plots'''

    # get system parameters
    parameters = get_parameters_size()[-1]
    parameters.num_partitions = 6700

    # set arithmetic complexity
    l = math.log2(parameters.num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    df = heuristic_fun(parameters)
    samples_bdc = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=partial(
            complexity.partitioned_encode_complexity,
            algorithm='gen',
        ),
        reduce_complexity_fun=partial(
            complexity.partitioned_reduce_complexity,
            algorithm='bm',
        )
    )
    np.save('samples_bdc', samples_bdc)
    cdf_bdc = simulation.cdf_from_samples(samples_bdc)
    print("bdc mean", samples_bdc.mean())

    df = rs_fun(parameters)
    samples_rs = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=partial(
            complexity.partitioned_encode_complexity,
            partitions=1,
            algorithm='fft',
        ),
        reduce_complexity_fun=partial(
            complexity.partitioned_reduce_complexity,
            partitions=1,
            algorithm='fft',
        ),
    )
    cdf_rs = simulation.cdf_from_samples(samples_rs)

    # LT codes
    # cdf_lt, samples_lt = get_lt_cdf(parameters, partitioned=False)
    # cdf_lt_partitioned, samples_lt_partitioned = get_lt_cdf(parameters, partitioned=True)
    cdf_lt_partitioned = get_lt_cdf(parameters, partitioned=True)

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
    t = np.linspace(samples_uncoded.min(), samples_rs.max(), num=200)
    t_norm = t # / parameters.num_columns #  / complexity.map_complexity_unified(parameters)

    plt.figure()
    # plot 1-cdf with a log y axis
    plt.autoscale(enable=True)
    print(t_norm)
    print(cdf_lt_partitioned(t))
    plt.semilogy(
        t_norm, 1-cdf_bdc(t),
        heuristic_plot_settings['color']+'o-',
        label=r'BDC, Heuristic',
    )
    # plt.semilogy(
    #     t_norm, 1-cdf_lt(t),
    #     lt_plot_settings['color']+':',
    #     label='LT',
    # )
    plt.semilogy(
        t_norm, 1-cdf_lt_partitioned(t),
        lt_partitioned_plot_settings['color']+'^-',
        label='LT, Partitioned',
    )
    plt.semilogy(
        t_norm, 1-cdf_uncoded(t),
        uncoded_plot_settings['color']+'-',
        markevery=0.2,
        label='UC',
    )
    plt.semilogy(
        t_norm, 1-cdf_rs(t),
        rs_plot_settings['color']+'d--',
        markevery=0.2,
        label='Unified',
    )
    plt.legend(
        numpoints=1,
        shadow=True,
        # labelspacing=0,
        # columnspacing=0.05,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.ylabel(r'$\Pr(\rm{Delay} > t)$')
    plt.xlabel(r'$t$')
    # plt.ylim(1e-15, 1)
    # plt.xlim(0, 2e6)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./plots/tcom/deadline.pdf', dpi='figure', bbox_inches='tight')
    plt.show()
    return


    # plot 1-cdf normalized
    plt.figure()
    # normalize = 1-cdf_uncoded(t)
    # plt.semilogy(t, (1-cdf_bdc(t))/normalize, label='bdc')
    # plt.semilogy(t, (1-cdf_rs(t))/normalize, label='unified')
    # plt.legend()
    # plt.grid()

    # plot the empiric and fitted cdf's
    plt.autoscale(enable=True)
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
        label='UC',
    )

    plt.plot(t, cdf_lt(t), lt_plot_settings['color'])
    plt.hist(
        samples_lt, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_plot_settings['color'],
        label='LT',
    )

    plt.plot(t, cdf_lt_partitioned(t), lt_partitioned_plot_settings['color'])
    plt.hist(
        samples_lt_partitioned, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_partitioned_plot_settings['color'],
        label='LT, Partitioned',
    )

    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        loc='best',
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.ylabel(r'PDF')
    plt.xlabel(r'$t$')
    plt.tight_layout()
    plt.grid()
    # plt.savefig('./plots/tcom/deadline_cdf.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def fuck():
    # samples = np.load('./lt_samples.npy')
    # plt.hist(
    #     samples, bins=100, density=True, cumulative=True,
    #     histtype='stepfilled',
    #     alpha=0.3,
    #     color=lt_plot_settings['color'],
    #     label='LT',
    # )
    # cdf = simulation.cdf_from_samples(samples)
    # t = np.linspace(samples.min(), samples.max(), 200)
    # plt.plot(t, cdf(t), lt_plot_settings['color'])

    samples_bdc = np.load('./samples_bdc.npy')
    plt.hist(
        samples_bdc, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=heuristic_plot_settings['color'],
        label='BDC',
    )
    cdf = simulation.cdf_from_samples(samples_bdc)
    t = np.linspace(samples_bdc.min(), samples_bdc.max(), 200)
    plt.plot(t, cdf(t), heuristic_plot_settings['color'])

    samples = np.load('./lt_samples_partitioned.npy')
    np.sort(samples)
    i = np.searchsorted(samples, samples_bdc.max())
    samples = samples[:i]
    plt.hist(
        samples, bins=200, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_partitioned_plot_settings['color'],
        label='LT, Partitioned',
    )
    cdf = simulation.cdf_from_samples(samples, n=1)
    t = np.linspace(samples.min(), samples.max(), 200)
    plt.plot(t, cdf(t), lt_partitioned_plot_settings['color'])

    # a=32.81815358407226
    # loc=289874.5292694412
    # scale=660.0449511451114
    # mean = scipy.stats.gamma.mean(a, loc=loc, scale=scale)
    # var = scipy.stats.gamma.var(a, loc=loc, scale=scale)
    # print(samples.mean(), mean)
    # print(samples.var(), var)

    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # lt_parameters()
    # lt_plots()
    # partition_plot()
    # size_plot()
    # workload_plot()
    # deadline_plot()
    # fuck()

    print(performance)
    plt.hist(
        performance['servers'], bins=200, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_partitioned_plot_settings['color'],
        label='LT, Partitioned',
    )
    plt.show()
    print(samples)
