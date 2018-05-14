import logging
import model
import plot
import rateless
import complexity
import simulation
import matplotlib.pyplot as plt
import pyrateless

from functools import partial
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
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
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
    # load/delay as function of system size
    plot.load_delay_plot(
        [rs],
        [rs_plot_settings],
        'num_partitions',
        xlabel=r'$T$',
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
    )
    plt.show()
    return

def size_plot():
    parameters = get_parameters_size()[:-2] # -2
    [print(p.num_source_rows) for p in parameters]
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
    
    plot.load_delay_plot(
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
        normalize=uncoded,
        legend='load',
        ncol=2,
        show=False,
        xlim_bot=(6, 201),
    )

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
    
# size_parameters = plot.get_parameters_size_2()[0:-2] # -2

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # lt_parameters()
    # partition_plot()
    size_plot()
    
