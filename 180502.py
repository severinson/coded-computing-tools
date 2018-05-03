import math
import logging
import matplotlib.pyplot as plt
import model
import complexity
import simulation
import plot

from scipy.special import comb as nchoosek
from evaluation.binsearch import SampleEvaluator
from evaluation import analytic
from solvers.heuristicsolver import HeuristicSolver
from functools import partial

# pyplot setup
plt.style.use('ggplot')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['figure.dpi'] = 200

# performance evaluators
sample_100 = SampleEvaluator(num_samples=100)
sample_1000 = SampleEvaluator(num_samples=1000)

# evaluator functions
uncoded_fun = partial(
    simulation.simulate,
    directory='./results/Uncoded/',
    samples=1,
    parameter_eval=analytic.uncoded_performance,
)
heuristic_fun = partial(
    simulation.simulate,
    directory='./results/Heuristic/',
    samples=1,
    solver=HeuristicSolver(),
    assignment_eval=sample_100,
)
rs_fun = partial(
    simulation.simulate,
    directory='./results/RS/',
    samples=1,
    parameter_eval=analytic.mds_performance,
)

# plot settings
heuristic_plot_settings = {
    'label': r'BDC',
    'color': 'r',
    'marker': 'd-'}
rs_plot_settings = {
    'label': r'RS BM',
    'color': 'k',
    'marker': 'v-'}

def get_parameters_workload(num_servers, W=1e8, num_partitions=None, code_rate=2/3, muq=2, tol=0.05):
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

    # assume num_outputs is a constant multiple of q
    # root = math.sqrt(W / muq)
    # num_source_rows = round(root)
    # num_outputs = q
    # num_columns = num_source_rows

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

def get_parameters_constant_workload():
    l = list()
    W_target = 1e8
    min_source_rows = 0 # ensure number of source rows is always increasing
    for i in range(6, 400):
        try:
            m = get_parameters_workload(i, W=W_target)
        except ValueError as err:
            continue
        if m.num_source_rows <= min_source_rows:
            continue
        min_source_rows = m.num_source_rows
        l.append(m)
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

def main():
    parameters = get_parameters_constant_workload()[:-2]
    # return
    # [print(p) for p in parameters]

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
            partitions=1
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
        ),
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    plot.load_delay_plot(
        [heuristic,
         rs],
        [heuristic_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded,
        show=False,
        # xlim_bot=(6, 201),
        # ylim_top=(0.4, 0.7),
        # ylim_bot=(0.5, 4.5),
    )
    # plt.savefig("./plots/180419/fft_8_load_delay.png", dpi='figure')

    plot.encode_decode_plot(
        [heuristic,
         rs],
        [heuristic_plot_settings,
         rs_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=None,
        show=False,
        # xlim_bot=(6, 201),
        # ylim_top=(0, 0.3),
        # ylim_bot=(0, 0.001),
    )
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
