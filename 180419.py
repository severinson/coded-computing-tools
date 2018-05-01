import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import model
import simulation
import complexity
import rateless
import plot
import stats

from math import log2
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

def get_parameters_size_10():
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
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def get_parameters_size_20():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 20
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
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def r10_pdf(overhead_levels, num_inputs=None, **kwargs):
    '''approximate completion PDF for R10 codes'''
    assert num_inputs is not None

    overhead_levels = np.fromiter(overhead_levels, dtype=float)
    overhead_levels.sort()
    if overhead_levels[1]-1 >= 0:
        overhead_levels -= 1

    pdf = np.zeros(len(overhead_levels))

    # evaluate the CDF at discrete points and extrapolate between them
    pf = [1e-1, 1e-2, 1e-3]
    a = [5.43476844e-03, 9.24066372e-03, 1.36168053e-02]

    # average of 1e-1 and 1e-2 values since the distance to the measured pf is
    # much smaller for these points.
    b = (8.09230267e-06 + 8.01977332e-06) / 2
    f = lambda t,a,b: a+b*t
    eps = np.finfo(float).eps

    # compute the overhead required for a failure probability of 1e-1, 1e-2,
    # 1e-3, and close to 0.
    y1 = 1-1e-1
    y2 = 1-1e-2
    y3 = 1-1e-3
    y4 = 1
    x1 = f(num_inputs, a[0], b) # 1e-1
    x2 = f(num_inputs, a[1], b) # 1e-2
    x3 = f(num_inputs, a[2], b) # 1e-3
    x4 = 2*x3 # assume failure probability 0 here

    # find the break points
    i1 = np.searchsorted(overhead_levels, x1)
    i2 = np.searchsorted(overhead_levels, x2)
    i3 = np.searchsorted(overhead_levels, x3)
    i4 = np.searchsorted(overhead_levels, x4)

    # assign the derivative of the cdf
    pdf[i1-1] = y1
    pdf[i1:i2] = (y2-y1) / (i2-i1)
    pdf[i2:i3] = (y3-y2) / (i3-i2)
    pdf[i3:i4] = (y4-y3) / (i4-i3)
    pdf[i4:] = 0
    print('pdf', pdf)
    assert np.allclose(pdf.sum(), 1), "sum of pdf should be 1, but is {}".format(pdf.sum())
    return pdf

def rq_pdf(overhead_levels, num_inputs=None, **kwargs):
    '''approximate completion PDF for RQ codes'''
    assert num_inputs is not None
    overhead_levels = np.fromiter(overhead_levels, dtype=float)
    overhead_levels.sort()
    if overhead_levels[1]-1 >= 0:
        overhead_levels -= 1
    cdf = np.zeros(len(overhead_levels))
    cdf[overhead_levels >= 0] = 1-1/100
    cdf[overhead_levels >= 1/num_inputs] = 1-1/1e4
    cdf[overhead_levels >= 2/num_inputs] = 1-1/1e6
    cdf[overhead_levels >= 3/num_inputs] = 1 # assume zero failure probability here
    pdf = np.zeros(len(overhead_levels))
    pdf[1:] = np.diff(cdf)
    pdf[0] = cdf[0]
    assert np.allclose(pdf.sum(), 1), "sum of pdf should be 1, but is {}".format(pdf.sum())
    return pdf

def R10_decoding_complexity(parameters):
    assert isinstance(parameters, model.SystemParameters)
    # linear regression of complexity at failure probability 1e-1
    a = 2.22413913e+02
    b = 3.64861777e-02

    # complexity as fp=1e-1 as a function of num_inputs
    f = lambda t,a,b: a+b*t
    return f(parameters.num_source_rows, a, b)

def RQ_decoding_complexity(parameters):
    assert isinstance(parameters, model.SystemParameters)
    # TODO: fix
    # linear regression of complexity at failure probability 1e-1
    a = 2.22413913e+02
    b = 3.64861777e-02

    # complexity as fp=1e-1 as a function of num_inputs
    f = lambda t,a,b: a+b*t
    return f(parameters.num_source_rows, a, b)

def rateless_evaluate(parameters, code='R10', pdf_fun=None, cachedir=None, partitioned=False):
    '''evaluate LT code performance.

    args:

    parameters: system parameters.

    returns: dict with performance results.

    '''
    assert isinstance(parameters, model.SystemParameters)
    assert code in ['R10', 'RQ']
    result = dict()

    # we encode each column of the input matrix separately
    if code == 'R10':
        result['encoding_multiplications'] = R10_encoding_complexity(parameters)
    elif code == 'RQ':
        result['encoding_multiplications'] = RQ_encoding_complexity(parameters)
    result['encoding_multiplications'] *= parameters.num_columns

    # we decode each output vector separately
    if code == 'R10':
        result['decoding_multiplications'] = R10_decoding_complexity(parameters)
    elif code == 'RQ':
        result['decoding_multiplications'] = RQ_decoding_complexity(parameters)
    result['decoding_multiplications'] *= parameters.num_outputs

    # each coded row is encoded by server_storage * q = muq servers.
    result['encoding_multiplications'] *= parameters.muq

    # compute encoding delay
    result['encode'] = stats.order_mean_shiftexp(
        parameters.num_servers,
        parameters.num_servers,
        parameter=result['encoding_multiplications'] / parameters.num_servers,
    )

    # compute decoding delay
    result['reduce'] = stats.order_mean_shiftexp(
        parameters.q,
        parameters.q,
        parameter=result['decoding_multiplications'] / parameters.q,
    )

    # simulate the map phase load/delay. this simulation takes into account the
    # probability of decoding at various levels of overhead.
    simulated = rateless.performance_integral(
        parameters=parameters,
        num_inputs=parameters.num_source_rows,
        target_overhead=1,
        mode=0,
        delta=0,
        pdf_fun=pdf_fun,
        max_overhead=1.1,
        cachedir=cachedir,
    )
    result['delay'] = simulated['delay']
    result['load'] = simulated['load']

    return result

# Setup the evaluators
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
)
r10_fun = partial(
    simulation.simulate,
    directory='./results/R10/',
    samples=1,
    parameter_eval=partial(
        rateless_evaluate,
        code='R10',
        pdf_fun=r10_pdf,
        cachedir='./results/R10',
    ),
)
rq_fun = partial(
    simulation.simulate,
    directory='./results/RQ/',
    samples=1,
    parameter_eval=partial(
        rateless_evaluate,
        code='RQ',
        pdf_fun=rq_pdf,
    ),
)
rs_fun = partial(
    simulation.simulate,
    directory='./results/RS/',
    samples=1,
    parameter_eval=analytic.mds_performance,
)

heuristic_plot_settings = {
    'label': r'BDC',
    'color': 'r',
    'marker': 'd-'}
heuristic_fft_plot_settings = {
    'label': r'BDC FFT',
    'color': 'g',
    'marker': 's-'}
r10_plot_settings = {
    'label': r'R10',
    'color': 'g',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
rq_plot_settings = {
    'label': r'RQ',
    'color': 'b',
    'marker': 'd-',
    'linewidth': 4,
    'size': 2}
lt_plot_settings = {
    'label': r'LT',
    'color': 'c',
    'marker': 'v-',
    'linewidth': 4,
    'size': 2}
rs_plot_settings = {
    'label': r'RS BM',
    'color': 'k',
    'marker': 'v-'}
rs_fft_plot_settings = {
    'label': r'RS FFT',
    'color': 'k',
    'marker': '-o'}

def complexity_from_df(df):
    '''assuming GF256 source symbols'''
    # each binary operations means adding two GF256 symbols
    df['complexity'] = 8*df['b']

    # each GF256-addition means multiplying by a GF256-symbols and adding then
    # adding.
    df['complexity'] += (8*log2(8)+8)*df['f']

    return df

def R10_encoding_complexity(p, partitioned=True):
    '''return the encoding complexity per source matrix column'''
    assert isinstance(p, model.SystemParameters)
    df = pd.read_csv("./R10.csv")
    complexity_from_df(df)
    K = p.num_source_rows
    if partitioned:
        K /= p.rows_per_batch
    i = df['K'].searchsorted(K)
    if i >= len(df):
        return np.inf
    K1, K2 = df['K'].values[i-1], df['K'].values[i]
    c1, c2 = df['complexity'].values[i-1], df['complexity'].values[i]
    precode = ((K-K1)*c1 + (K2-K)*c2) / (K2-K1)
    if partitioned:
        precode *= p.rows_per_batch
    outer = p.num_coded_rows * 4.631353378295898 * 8
    return float(precode + outer)

def RQ_encoding_complexity(p, partitioned=True):
    '''return the encoding complexity per source matrix column'''
    assert isinstance(p, model.SystemParameters)
    df = pd.read_csv("./RQ.csv")
    complexity_from_df(df)
    K = p.num_source_rows
    if partitioned:
        K /= p.rows_per_batch
    i = df['K'].searchsorted(K)
    if i >= len(df):
        return np.inf
    K1, K2 = df['K'].values[i-1], df['K'].values[i]
    c1, c2 = df['complexity'].values[i-1], df['complexity'].values[i]
    precode = ((K-K1)*c1 + (K2-K)*c2) / (K2-K1)
    if partitioned:
        precode *= p.rows_per_batch
    outer = p.num_coded_rows * 7.152566 * 8
    return float(precode + outer)

def precode_complexity_plot():
    R10 = pd.read_csv("./R10.csv")
    complexity_from_df(R10)
    RQ = pd.read_csv("./RQ.csv")
    complexity_from_df(RQ)
    print(R10)
    print(RQ)

    plt.figure()
    plt.semilogx(R10['K'], R10['complexity'] / R10['K'], label='R10')
    plt.semilogx(RQ['K'], RQ['complexity'] / RQ['K'], label='RQ')
    plt.title("Raptor Precode Complexity")
    plt.xlabel('Source Symbols')
    plt.ylabel('Complexity Per Source Symbol')
    plt.tight_layout()
    plt.xlim(0, 1e5)
    plt.ylim(0, 600)
    plt.savefig('./plots/180419/precode_complexity.png', dpi='figure')
    plt.show()
    return

def encoding_complexity_plot():
    parameters = get_parameters_size_10()
    x = [p.num_source_rows for p in parameters]
    plt.figure()

    R10 = [R10_encoding_complexity(p) for p in parameters]
    RQ = [RQ_encoding_complexity(p) for p in parameters]
    plt.semilogy(x, R10, label='R10 Partitioned')
    plt.semilogy(x, RQ, label='RQ Partitioned')

    R10 = [R10_encoding_complexity(p, partitioned=False) for p in parameters]
    RQ = [RQ_encoding_complexity(p, partitioned=False) for p in parameters]
    plt.semilogy(x, R10, label='R10')
    plt.semilogy(x, RQ, label='RQ')

    plt.title("Raptor Encoding Complexity")
    plt.xlabel('Source Symbols')
    plt.ylabel('Complexity')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 140000)
    plt.ylim(1e5, 1e10)
    plt.savefig('./plots/180419/encode_complexity.png', dpi='figure')
    plt.show()

def partitioning_plot():
    # num_inputs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    # overhead_levels = np.linspace(0, 0.1)
    # plt.figure()
    # for K in num_inputs:
    #     plt.plot(
    #         overhead_levels,
    #         r10_pdf(overhead_levels, num_inputs=K),
    #         label="R10 $K={}$".format(K),
    #     )
    #     plt.plot(
    #         overhead_levels,
    #         RQ_pdf(overhead_levels, num_inputs=K),
    #         label="RQ $K={}$".format(K),
    #     )
    # plt.title("R10 Completion PDF")
    # plt.xlabel("Relative Overhead")
    # plt.ylabel("Probability")
    # plt.xlim((0, 0.1))
    # plt.ylim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # return

    parameters = plot.get_parameters_partitioning()
    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    r10 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=r10_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    rq = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=rq_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )

    plot.load_delay_plot(
        [heuristic,
         r10,
         rq],
        [heuristic_plot_settings,
         r10_plot_settings,
         rq_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded,
        show=False,
        xlim_bot=(10, 3000),
        ylim_top=(0.51, 0.58),
        ylim_bot=(0, 200),
    )
    plt.savefig("./plots/180419/load_delay.png", dpi='figure')

    plot.encode_decode_plot(
        [heuristic,
         r10,
         rq],
        [heuristic_plot_settings,
         r10_plot_settings,
         rq_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=None,
        show=False,
        xlim_bot=(2, 3000),
        ylim_top=(0.2, 1),
        ylim_mid=(0, 0.00035),
        ylim_bot=(0, 0.8),
    )
    # plt.savefig("./plots/180328/phases.png", dpi='figure')
    plt.show()

def size_plot():
    parameters = get_parameters_size_20()[:-1]
    # parameters = plot.get_parameters_partitioning_2()

    uncoded = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=uncoded_fun,
        map_complexity_fun=complexity.map_complexity_uncoded,
        encode_delay_fun=lambda x: 0,
        reduce_delay_fun=lambda x: 0,
    )
    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    r10 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=r10_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    rq = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=rq_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    parameters = get_parameters_size_10()[:-1]
    parameters[-2:] = get_parameters_size_20()[-3:-1]
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )

    plot.load_delay_plot(
        [heuristic,
         lt,
         r10,
         rq],
        [heuristic_plot_settings,
         lt_plot_settings,
         r10_plot_settings,
         rq_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0.4, 1),
        ylim_bot=(0.8, 2.4),
    )
    # plt.savefig("./plots/180309/load_delay.png")

    plot.encode_decode_plot(
        [heuristic,
         lt,
         r10,
         rq],
        [heuristic_plot_settings,
         lt_plot_settings,
         r10_plot_settings,
         rq_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=None,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0, 0.3),
        ylim_bot=(0, 0.001),
    )
    # plt.savefig("./plots/180309/encode_decode.png")

    plt.show()
    return

def rs_plot():
    parameters = get_parameters_size_20()
    [print(p) for p in parameters]

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
            algorithm='bm',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            partitions=1,
            algorithm='bm',
        ),
    )
    rs_fft = simulation.simulate_parameter_list(
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
    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    parameters = get_parameters_size_10()
    parameters[-3:] = get_parameters_size_20()[-3:]
    heuristic = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=partial(
            complexity.partitioned_encode_delay,
            algorithm='bm',
        ),
        reduce_delay_fun=partial(
            complexity.partitioned_reduce_delay,
            algorithm='bm',
        ),
    )
    heuristic_fft = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=heuristic_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=complexity.partitioned_encode_delay,
        reduce_delay_fun=complexity.partitioned_reduce_delay,
    )
    plot.load_delay_plot(
        [heuristic,
         heuristic_fft,
         # rs,
         rs_fft,
         lt],
        [heuristic_plot_settings,
         heuristic_fft_plot_settings,
         # rs_plot_settings,
         rs_fft_plot_settings,
         lt_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded,
        show=False,
        xlim_bot=(6, 201),
        ylim_top=(0.4, 0.7),
        ylim_bot=(0.5, 4.5),
    )
    plt.savefig("./plots/180419/fft_8_load_delay.png", dpi='figure')

    plot.encode_decode_plot(
        [heuristic,
         rs,
         rs_fft],
        [heuristic_plot_settings,
         rs_plot_settings,
         rs_fft_plot_settings],
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
    logging.basicConfig(level=logging.DEBUG)
    # partitioning_plot()
    # size_plot()
    rs_plot()

    # encoding_complexity_plot()
    # parameters = get_parameters_size_10()
    # precode_complexity_plot()
    # c = RQ_encoding_complexity(parameters[2])
    # print(c)
