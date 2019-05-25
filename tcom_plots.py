import math
import logging
import numpy as np
import scipy.stats
import pynumeric
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
from matplotlib2tikz import save as tikz_save
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
lt_fun_03 = partial(
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
lt_partitioned_fun_03 = partial(
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
    rerun=True,
)
lt_partitioned_fun_01 = partial(
    simulation.simulate,
    directory='./results/LT_1_1_partitioned/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1.1,
        target_failure_probability=1e-1,
        partitioned=True,
        cachedir='./results/LT_1_1_partitioned/overhead',
    ),
    rerun=True,
)

def get_parameters_tradeoff(alpha=0.01, all_T=False):
    '''Get a list of parameters for the load-vs-delay plot.'''
    num_outputs = 840
    num_servers = 14
    server_storage = 1/2
    parameters = list()
    # num_source_rows = 352716
    num_source_rows = 50000
    for q in range(1, num_servers):
        muq = int(round(server_storage*q))
        num_coded_rows = int(round(num_source_rows * num_servers / q))
        num_batches = nchoosek(num_servers, muq)
        rows_per_batch = int(round(num_coded_rows / num_batches))
        # num_columns = int(round(alpha*num_source_rows))

        try:
            par = model.SystemParameters(
                rows_per_batch=rows_per_batch,
                num_servers=num_servers,
                q=q,
                num_outputs=num_outputs,
                server_storage=server_storage,
                num_partitions=rows_per_batch,
                num_columns=int(round(alpha*num_source_rows)),
            )
            print(par)
            print(abs(1-par.num_source_rows / num_source_rows)*100)
        except ValueError:
            continue

        if not all_T:
            parameters.append(par)
            continue

        for T in range(par.rows_per_batch, par.num_source_rows+1):
            try:
                dct = par.asdict()
                dct['num_partitions'] = T
                par = model.SystemParameters.fromdct(dct)
            except ValueError as err:
                continue
            parameters.append(par)

    return parameters

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
    num_columns = 0.01
    num_outputs_factor = 500
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200]
    num_partitions = [500, 1000, 1750, 3400, 3000, 4200, 6700]
    for (servers, T) in zip(num_servers, num_partitions):
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor,
        )
        dct = par.asdict()
        dct['num_partitions'] = T
        par = model.SystemParameters.fromdct(dct)
        parameters.append(par)
    return parameters

def get_parameters_deadline():
    p = model.SystemParameters.fixed_complexity_parameters(
        rows_per_server=2000,
        rows_per_partition=10,
        min_num_servers=200,
        code_rate=2/3,
        muq=2,
        num_columns=0.01,
        num_outputs_factor=500,
    )
    dct = p.asdict()
    dct['num_partitions'] = 6700
    p = model.SystemParameters.fromdct(dct)
    return p

def get_parameters_size_partitions():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = 0.01
    num_outputs_factor = 500
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
                # dct['num_columns'] = 100*par.num_columns
                p = model.SystemParameters.fromdct(dct)
            except ValueError as err:
                continue
            parameters.append(p)
    return parameters

def get_parameters_workload(num_servers, W=1e8, num_partitions=None,
                            code_rate=2/3, muq=2, alpha=0.01, tol=0.05):
    '''Get a list of parameters for the size plot.'''
    q = code_rate*num_servers
    server_storage = muq / q

    # assume num_outputs and num_columns is a constant factor of
    # num_source_rows
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
    for i in range(1, 20):
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

def lt_parameters(tfp=None, to=None, partitioned=False):
    print('getting LT code parameters')
    parameters = [get_parameters_deadline()]
    for p in parameters:
        print(p)
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

    # set arithmetic complexity
    l = math.log2(parameters[-1].num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    # since caching assumes n/N remains constant we have to re-run
    # each time.
    rerun = True
    lt_3_1_fun = partial(
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
    lt_37_1_fun = partial(
        simulation.simulate,
        directory='./results/LT_37_1/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.37,
            target_failure_probability=1e-1,
            cachedir='./results/LT_37_1/overhead',
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
            cachedir='./results/LT_3_3/overhead',
        ),
        rerun=rerun,
    )
    lt_37_3_fun = partial(
        simulation.simulate,
        directory='./results/LT_37_3/',
        samples=1,
        parameter_eval=partial(
            rateless.evaluate,
            target_overhead=1.37,
            target_failure_probability=1e-3,
            cachedir='./results/LT_37_3/overhead',
        ),
        rerun=rerun,
    )

    lt_3_1 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_3_1_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_37_1 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_37_1_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_3_3 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_3_3_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_37_3 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_37_3_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    # run simulations
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
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )

    settings_3_1 = {
        'label': r'LT $(0.3, 10^{-1})$',
        'color': 'g',
        'marker': '^-',
        'markevery': 0.15}
    settings_37_1 = {
        'label': r'LT $(0.37, 10^{-1})$',
        'color': 'm',
        'marker': 's-',
        'markevery': 0.2}
    settings_3_3 = {
        'label': r'LT $(0.3, 10^{-3})$',
        'color': 'k',
        'marker': '^--',
        'markevery': 0.25}
    settings_37_3 = {
        'label': r'LT $(0.37, 10^{-3})$',
        'color': 'b',
        'marker': 's--',
        'markevery': 0.3}

    plot.load_delay_plot(
        [heuristic,
         lt_3_1,
         lt_3_3,
         lt_37_1,
         lt_37_3],
        [heuristic_plot_settings,
         settings_3_1,
         settings_3_3,
         settings_37_1,
         settings_37_3],
        'num_columns',
        xlabel=r'$n$',
        normalize=uncoded,
        ncol=2,
        xlim_bot=(parameters[0].num_columns, parameters[-1].num_columns),
        ylim_top=(0.5, 1.0),
        ylim_bot=(2.0, 2.30),
        show=False,
    )
    # plt.savefig('./plots/tcom/lt.pdf')
    # plt.savefig('./plots/tcom/lt.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/lt.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )
    plt.show()
    return

def partition_plot():
    parameters = get_parameters_partitioning()

    # set arithmetic complexity
    # l = math.log2(parameters[-1].num_coded_rows)
    l = math.ceil(math.log2(parameters[-1].num_coded_rows + 1))
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
    # plt.savefig('./plots/tcom/partitions.png', dpi='figure')
    # plt.savefig('./plots/tcom/partitions.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/partitions.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )

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
        ncol=1,
        show=False,
        vline=parameters[0].rows_per_batch,
        ylim_top=(0.5, 0.6),
        ylim_bot=(1.5, 4),
        xlim_bot=(2, 3000),
    )
    # plt.savefig('./plots/tcom/solvers_partitions.png', dpi='figure')
    # plt.savefig('./plots/tcom/solvers_partitions.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/solvers_partitions.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )

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
    print(heuristic)
    print(heuristic['num_partitions'])

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
        ylim_top=(0.4, 1.1),
        ylim_bot=(0.5, 2.5),
    )
    # plt.savefig('./plots/tcom/size.png', dpi='figure')
    # plt.savefig('./plots/tcom/size.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/size.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )

    plot.load_delay_plot(
        [heuristic,
         random],
        [heuristic_plot_settings,
         random_plot_settings],
        'num_servers',
        xlabel=r'$K$',
        normalize=uncoded,
        legend='load',
        ncol=1,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0.4, 0.7),
        ylim_bot=(0.5, 2.5),
    )
    # plt.savefig('./plots/tcom/solvers_size.png', dpi='figure')
    # plt.savefig('./plots/tcom/solvers_size.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/solvers_size.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )

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

    # # comparison printouts
    # key = 'overall_delay'
    # for df in [heuristic, lt, lt_partitioned, rs]:
    #     df['overall_delay'] /= uncoded['overall_delay']
    #     df['load'] /= uncoded['load']

    # print('heuristic-rs', 1-heuristic[key]/rs[key])
    # print('lt-heuristic', 1-lt[key]/heuristic[key])
    # print('ltp-heuristic', 1-lt_partitioned[key]/heuristic[key])

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

    print(heuristic_101)
    print(heuristic_101['num_partitions'])

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
    # plt.savefig('./plots/tcom/workload.png', dpi='figure')
    # plt.savefig('./plots/tcom/workload.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/workload.tex',
        figureheight='\\figureheightd',
        figurewidth='\\figurewidth'
    )

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

    plt.show()
    return

def get_lt_cdf(parameters, partitioned=False):
    '''sample the LT code delay distribution and return a
    gamma-distributed CDF fit to the samples.

    Fit parameters for TCOM plot:
    Non-partitioned  a=25.9785443475378, loc=285150.2196003293, scale=4782.081148149096
    Partitioned a=14.18576693899206, loc=302693.1614997096, scale=6349.8236659845315

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert isinstance(partitioned, bool)
    if partitioned:
        # a=14.18576693899206
        # loc=302693.1614997096
        # scale=6349.8236659845315
        # return lambda t: scipy.stats.gamma.cdf(t, a, loc=loc, scale=scale)

        filename = "samples_lt_partitioned"
        num_partitions = parameters.rows_per_batch
        target_overhead = 1.335
        cachedir='./results/LT_335_1_partitioned/overhead'
        # order_values, order_probabilities = rateless.order_pdf(
        #     parameters=parameters,
        #     target_overhead=1.3,
        #     target_failure_probability=1e-1,
        #     partitioned=partitioned,
        # )

    else:
        # a=25.9785443475378
        # loc=285150.2196003293
        # scale=4782.081148149096
        # return lambda t: scipy.stats.gamma.cdf(t, a, loc=loc, scale=scale)

        filename = "samples_lt"
        num_partitions = 1
        target_overhead = 1.335
        cachedir='./results/LT_335_1/overhead'

    num_inputs = int(parameters.num_source_rows / num_partitions)
    overheads = rateless.lt_success_samples(
        10000,
        target_overhead=target_overhead,
        num_inputs=num_inputs,
        mode=num_inputs-2,
        delta=0.9999999701976676,
    )
    df = overhead.performance_from_overheads(
        overheads,
        parameters=parameters,
        design_overhead=target_overhead,
    )
    df.to_csv(filename + '.csv')

    encoding_complexity = rateless.lt_encoding_complexity(
        num_inputs=num_inputs,
        failure_prob=1e-1,
        target_overhead=target_overhead,
        code_rate=parameters.q/parameters.num_servers,
    )
    encoding_complexity *= parameters.num_columns
    encoding_complexity *= num_partitions
    encoding_complexity *= parameters.muq

    decoding_complexity = rateless.lt_decoding_complexity(
        num_inputs=num_inputs,
        failure_prob=1e-1,
        target_overhead=target_overhead,
    )
    decoding_complexity *= num_partitions
    decoding_complexity *= parameters.num_outputs

    print('LT encoding/decoding complexity: {}/{}'.format(
        encoding_complexity,
        decoding_complexity,
    ))
    samples = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=lambda x: encoding_complexity,
        reduce_complexity_fun=lambda x: decoding_complexity,
    )
    np.save(filename, samples)
    print("mean delay is", samples.mean())

    return simulation.cdf_from_samples(samples), samples

def deadline_plot_old():
    '''deadline plots'''

    # get system parameters
    parameters = get_parameters_deadline()

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
    print("BDC mean", samples_bdc.mean())

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
    cdf_lt, samples_lt = get_lt_cdf(parameters, partitioned=False)
    cdf_lt_partitioned, samples_lt_partitioned = get_lt_cdf(parameters, partitioned=True)
    # cdf_lt = get_lt_cdf(parameters, partitioned=False)
    # cdf_lt_partitioned = get_lt_cdf(parameters, partitioned=True)

    df = uncoded_fun(parameters)
    samples_uncoded = simulation.delay_samples(
        df,
        parameters=parameters,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_complexity_fun=False,
        reduce_complexity_fun=False,
    )
    np.save('samples_uncoded', samples_uncoded)
    print("Uncoded mean", samples_uncoded.mean())
    cdf_uncoded = simulation.cdf_from_samples(samples_uncoded)

    # find points to evaluate the cdf at
    # t = np.linspace(
    #     1.5 * complexity.map_complexity_unified(parameters),
    #     5 * complexity.map_complexity_unified(parameters),
    # )
    # t = np.linspace(samples_uncoded.min(), 12500)
    t = np.linspace(0, 2*samples_uncoded.max(), num=200)
    t_norm = t # / parameters.num_columns #  / complexity.map_complexity_unified(parameters)

    plt.figure()
    # plot 1-cdf with a log y axis
    plt.autoscale(enable=True)
    plt.semilogy(
        t_norm, 1-cdf_bdc(t),
        heuristic_plot_settings['color']+'o-',
        markevery=0.2,
        label=r'BDC, Heuristic',
    )
    plt.semilogy(
        t_norm, 1-cdf_lt(t),
        lt_plot_settings['color']+'v-',
        markevery=0.2,
        label='LT',
    )
    plt.semilogy(
        t_norm, 1-cdf_lt_partitioned(t),
        lt_partitioned_plot_settings['color']+'^-',
        markevery=0.2,
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
    plt.ylim(1e-14, 1.13)
    plt.xlim(2.5e3, 5.5e3)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./plots/tcom/deadline_500.pdf', dpi='figure', bbox_inches='tight')
    # plt.show()
    # return

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

def tradeoff_plot():
    parameters = get_parameters_tradeoff()
    parameters_all_T = get_parameters_tradeoff(all_T=True)

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
        parameter_list=parameters_all_T,
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
        parameter_list=parameters_all_T,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='fft',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='fft',
        ),
    )

    # filter out rows with load over that of the RS code
    heuristic_110 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.1, :
    ]

    # filter out rows with load more than 1% over that of the RS code
    heuristic_101 = heuristic.loc[
        heuristic['load']/rs_all_t['load'] <= 1.01, :
    ]

    # find the optimal partitioning level for each number of servers
    heuristic_110 = heuristic_110.loc[
        heuristic_110.groupby("q")["overall_delay"].idxmin(), :
    ]
    heuristic_110.reset_index(inplace=True)
    heuristic_101 = heuristic_101.loc[
        heuristic_101.groupby("q")["overall_delay"].idxmin(), :
    ]
    heuristic_101.reset_index(inplace=True)

    lt = simulation.simulate_parameter_list(
        parameter_list=parameters[:-1],
        simulate_fun=lt_fun_03,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned_01 = simulation.simulate_parameter_list(
        parameter_list=parameters[1:],
        simulate_fun=lt_partitioned_fun_01,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_partitioned_03 = simulation.simulate_parameter_list(
        parameter_list=parameters[:-1],
        simulate_fun=lt_partitioned_fun_03,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    heuristic_110['overall_delay'] /= uncoded['overall_delay']
    heuristic_110['load'] /= uncoded['load']
    heuristic_110['encode'] /= uncoded['overall_delay']
    heuristic_110['reduce'] /= uncoded['overall_delay']
    heuristic_101['overall_delay'] /= uncoded['overall_delay']
    heuristic_101['load'] /= uncoded['load']
    heuristic_101['encode'] /= uncoded['overall_delay']
    heuristic_101['reduce'] /= uncoded['overall_delay']
    lt['overall_delay'] /= uncoded['overall_delay']
    lt['load'] /= uncoded['load']
    lt['encode'] /= uncoded['overall_delay']
    lt['reduce'] /= uncoded['overall_delay']
    lt_partitioned_01['overall_delay'] /= uncoded['overall_delay']
    lt_partitioned_01['load'] /= uncoded['load']
    lt_partitioned_01['encode'] /= uncoded['overall_delay']
    lt_partitioned_01['reduce'] /= uncoded['overall_delay']
    lt_partitioned_03['overall_delay'] /= uncoded['overall_delay']
    lt_partitioned_03['load'] /= uncoded['load']
    lt_partitioned_03['encode'] /= uncoded['overall_delay']
    lt_partitioned_03['reduce'] /= uncoded['overall_delay']
    rs['overall_delay'] /= uncoded['overall_delay']
    rs['load'] /= uncoded['load']
    rs['encode'] /= uncoded['overall_delay']
    rs['reduce'] /= uncoded['overall_delay']
    plt.figure()
    plt.plot(
        heuristic_101['overall_delay'],
        heuristic_101['load'],
        heuristic_plot_settings['color']+heuristic_plot_settings['marker'],
        label='BDC, Heuristic, $1$\%',
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.plot(
        heuristic_110['overall_delay'],
        heuristic_110['load'],
        'ms-',
        label='BDC, Heuristic, $10$\%',
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.plot(
        lt_partitioned_03['overall_delay'],
        lt_partitioned_03['load'],
        lt_partitioned_plot_settings['color']+lt_partitioned_plot_settings['marker'],
        label='LT, Partitioned',
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.plot(
        rs['overall_delay'],
        rs['load'],
        rs_plot_settings['color']+rs_plot_settings['marker'],
        label='Unified',
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.margins(y=0.1)
    plt.xlim(2, 5.5)
    plt.ylim(0.1, 0.9)
    plt.grid()
    plt.xlabel('$D$')
    plt.ylabel('$L$')
    # plt.plot(heuristic['encode'], heuristic['load'], label='Encode BDC, Heuristic')
    # plt.plot(rs['encode'], rs['load'], label='Encode Unified')
    # plt.plot(heuristic['reduce'], heuristic['load'], label='Reduce BDC, Heuristic')
    # plt.plot(rs['reduce'], rs['load'], label='Reduce Unified')
    plt.legend()
    # plt.savefig('./plots/tcom/tradeoff.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/tradeoff.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
    )
    plt.show()
    return

def hist_from_samples():
    '''used for loading delay samples and plotting a histogram'''
    samples = np.load('./lt_samples.npy')
    print("LT mean", samples.mean())
    plt.hist(
        samples, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_plot_settings['color'],
        label='LT',
    )
    cdf = simulation.cdf_from_samples(samples)
    t = np.linspace(samples.min(), samples.max(), 200)
    plt.plot(t, cdf(t), lt_plot_settings['color'])

    samples_bdc = np.load('./samples_bdc.npy')
    print("BDC mean", samples_bdc.mean())
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
    print("LT partitioned mean", samples.mean())
    plt.hist(
        samples, bins=200, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_partitioned_plot_settings['color'],
        label='LT, Partitioned',
    )
    cdf = simulation.cdf_from_samples(samples)
    t = np.linspace(samples.min(), samples.max(), 200)
    plt.plot(t, cdf(t), lt_partitioned_plot_settings['color'])

    plt.show()
    return

def lt_distribution(parameters,
                    num_samples=1000,
                    target_overhead=None,
                    target_failure_probability=None,
                    partitioned=False):
    if partitioned:
        num_partitions = parameters.rows_per_batch
    else:
        num_partitions = 1
    num_inputs = int(round(parameters.num_source_rows/num_partitions))

    # order_values, order_probabilities = rateless.order_pdf(
    #     parameters=parameters,
    #     target_overhead=target_overhead,
    #     target_failure_probability=target_failure_probability,
    #     partitioned=partitioned,
    #     num_overhead_levels=100,
    #     num_samples=100000,
    #     cachedir='./results/LT_335_6/overhead',
    # )
    order_values = [parameters.q]
    order_probabilities = [1]
    # print('LT order_values', order_values)
    # print('LT order_probabilities', order_probabilities)

    encoding_complexity = rateless.lt_encoding_complexity(
        num_inputs=num_inputs,
        failure_prob=target_failure_probability,
        target_overhead=target_overhead,
        code_rate=parameters.q/parameters.num_servers,
    )
    encoding_complexity *= parameters.num_columns
    encoding_complexity *= num_partitions
    encoding_complexity *= parameters.muq

    decoding_complexity = rateless.lt_decoding_complexity(
        num_inputs=num_inputs,
        failure_prob=target_failure_probability,
        target_overhead=target_overhead,
    )
    decoding_complexity *= num_partitions
    decoding_complexity *= parameters.num_outputs

    print('LT encoding/decoding complexity: {}/{}'.format(
        encoding_complexity,
        decoding_complexity,
    ))

    cdf, minv, maxv = simulation.infer_completion_cdf(
        parameters=parameters,
        order_values=order_values,
        order_probabilities=order_probabilities,
        num_samples=num_samples,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_complexity_fun=lambda x: encoding_complexity,
        reduce_complexity_fun=lambda x: decoding_complexity,
    )
    return cdf, minv, maxv

def deadline_plot(target_overhead=1.335,
                  num_samples=1000000):
    parameters = get_parameters_deadline()

    # set arithmetic complexity
    l = math.log2(parameters.num_coded_rows)
    complexity.ADDITION_COMPLEXITY = l/64
    complexity.MULTIPLICATION_COMPLEXITY = l*math.log2(l)

    # # sample the overhead required
    # overhead_samples = rateless.lt_success_samples(
    #     n,
    #     target_overhead=target_overhead,
    #     num_inputs=num_inputs,
    #     mode=num_inputs-2,
    #     delta=0.9999999701976676,
    # )
    # # np.save(filename, samples)
    # print('overhead samples', overhead_samples)

    # # convert overhead samples to number of servers needed
    # server_samples = overhead.performance_from_overheads(
    #     overhead_samples,
    #     parameters=parameters,
    #     design_overhead=target_overhead,
    # )
    # # server_samples.to_csv(filename + 'server_samples.csv')
    # print('server samples', server_samples)

    ## LT Codes ##
    # cdf_lt_3, minv_lt, maxv_lt = lt_distribution(
    #     parameters,
    #     num_samples=num_samples,
    #     partitioned=False,
    #     target_overhead=target_overhead,
    #     target_failure_probability=1e-3
    # )
    # cdf_lt_6, _, _ = lt_distribution(
    #     parameters,
    #     num_samples=num_samples,
    #     partitioned=False,
    #     target_overhead=target_overhead,
    #     target_failure_probability=1e-6,
    # )
    cdf_lt_9, minv_lt, maxv_lt = lt_distribution(
        parameters,
        num_samples=num_samples,
        partitioned=False,
        target_overhead=target_overhead,
        target_failure_probability=1e-9,
    )

    ## BDC Codes ##
    df = heuristic_fun(parameters)
    order_count = df['servers'].value_counts(normalize=True)
    order_count.sort_index(inplace=True)
    order_values = np.array(order_count.index)
    order_probabilities = order_count.values
    cdf_bdc, minv_bdc, maxv_bdc = simulation.infer_completion_cdf(
        parameters=parameters,
        order_values=order_values,
        order_probabilities=order_probabilities,
        num_samples=num_samples,
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

    ## RS Codes ##
    df = rs_fun(parameters)
    order_count = df['servers'].value_counts(normalize=True)
    order_count.sort_index(inplace=True)
    order_values = np.array(order_count.index)
    order_probabilities = order_count.values
    cdf_rs, minv_rs, maxv_rs = simulation.infer_completion_cdf(
        parameters=parameters,
        order_values=order_values,
        order_probabilities=order_probabilities,
        num_samples=num_samples,
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

    ## Uncoded ##
    df = uncoded_fun(parameters)
    order_count = df['servers'].value_counts(normalize=True)
    order_count.sort_index(inplace=True)
    order_values = np.array(order_count.index)
    order_probabilities = order_count.values
    cdf_uncoded, minv_uncoded, maxv_uncoded = simulation.infer_completion_cdf(
        parameters=parameters,
        order_values=order_values,
        order_probabilities=order_probabilities,
        num_samples=num_samples,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_complexity_fun=False,
        reduce_complexity_fun=False,
    )

    plt.figure()
    t = np.linspace(minv_lt, round(1.5*maxv_lt), 200)
    plt.semilogy(
        t,
        [1-cdf_bdc(x) for x in t],
        heuristic_plot_settings['color']+'o-',
        label='BDC, Heuristic',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
    )

    t_max = pynumeric.cnuminv(fun=cdf_lt_9, target=1-1e-9)
    t = np.linspace(minv_lt, t_max, 100)
    plt.semilogy(
        t,
        [1-cdf_lt_9(x) for x in t],
        'bv-',
        label='LT',
        # label='LT, $(1.335, 10^{-9})$',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
    )

    # t_max = pynumeric.cnuminv(fun=cdf_lt_6, target=1-1e-6)
    # t = np.linspace(minv_lt, t_max, 100)
    # plt.semilogy(
    #     t,
    #     [1-cdf_lt_6(x) for x in t],
    #     'c^-',
    #     label='LT, $(1.335, 10^{-6})$',
    #     markevery=0.2,
    # )

    # t_max = pynumeric.cnuminv(fun=cdf_lt_3, target=1-1e-3)
    # t = np.linspace(minv_lt, t_max, 100)
    # plt.semilogy(
    #     t,
    #     [1-cdf_lt_3(x) for x in t],
    #     'bv-',
    #     label='LT, $(1.335, 10^{-3})$',
    #     markevery=0.2,
    # )

    # plt.semilogy(t, [1-cdf_ltp(x) for x in t], label='LT, Partitioned')
    t = np.linspace(minv_lt, round(1.5*maxv_lt), 200)
    plt.semilogy(
        t,
        [1-cdf_rs(x) for x in t],
        rs_plot_settings['color']+'d--',
        label='Unified',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.semilogy(
        t,
        [1-cdf_uncoded(x) for x in t],
        uncoded_plot_settings['color'],
        label='UC',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
    )
    plt.ylabel(r'$\Pr(\rm{Delay} > t)$')
    plt.xlabel(r'$t$')
    plt.tight_layout()
    plt.ylim(1e-9, 1.0)
    plt.xlim(2.5e3, 5e3)
    plt.grid()
    plt.legend(
        numpoints=1,
        loc='best',
    )
    # plt.savefig('./plots/tcom/deadline.pdf', dpi='figure', bbox_inches='tight')
    tikz_save(
        './plots/tcom/deadline.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
    )
    plt.show()
    return

    lt_samples = np.load('./lt_samples.npy')
    print("LT mean", lt_samples.mean())
    plt.figure()
    plt.hist(
        lt_samples, bins=100, density=True, cumulative=True,
        histtype='stepfilled',
        alpha=0.3,
        color=lt_plot_settings['color'],
        label='LT',
    )
    t = np.linspace(lt_samples.min(), 2*lt_samples.max(), 200)
    plt.plot(t, [cdf_lt(x) for x in t], lt_plot_settings['color'])
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # [print(p) for p in get_parameters_N()]
    # lt_parameters(tfp=1e-9, to=1.335, partitioned=False)
    # lt_plots()
    # partition_plot()
    # size_plot()
    # workload_plot()
    deadline_plot()
    # tradeoff_plot()
    # hist_from_samples()
    # get_parameters_deadline()
