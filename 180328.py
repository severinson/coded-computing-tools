import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plot
import simulation
import complexity
import rateless
import model
import stats

from evaluation.binsearch import SampleEvaluator
from evaluation import analytic
from solvers.heuristicsolver import HeuristicSolver
from functools import partial

heuristic_plot_settings = {
    'label': r'BDC',
    'color': 'r',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
r10_plot_settings = {
    'label': r'R10',
    'color': 'g',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}

def r10_decoding_complexity(parameters):
    assert isinstance(parameters, model.SystemParameters)
    # linear regression of complexity at failure probability 1e-1
    a = 2.22413913e+02
    b = 3.64861777e-02

    # complexity as fp=1e-1 as a function of num_inputs
    f = lambda t,a,b: a+b*t
    return f(parameters.num_source_rows, a, b)

def r10_encoding_complexity(parameters):
    # each source symbol is part of 3 ldpc symbols.
    # so the total degree of all ldpc symbols is 3*num_source rows.
    # the degree of each hdpc symbol is about num_source_rows / 2
    assert isinstance(parameters, model.SystemParameters)
    num_ldpc = {
        4000: 131,
        6000: 173,
        14000: 311,
        34000: 607,
        54000: 877,
        84000: 1259,
        134000: 1861,
    }
    num_hdpc = {
        4000: 15,
        6000: 15,
        14000: 17,
        34000: 18,
        54000: 19,
        84000: 19,
        134000: 20,
    }
    if parameters.num_source_rows not in num_ldpc:
        raise ValueError("parameters.num_source rows must be in {}".format(num_ldpc))
    if parameters.num_source_rows not in num_hdpc:
        raise ValueError("parameters.num_source rows must be in {}".format(num_hdpc))

    m = parameters.num_source_rows
    ldpc = num_ldpc[parameters.num_source_rows]
    hdpc = num_hdpc[parameters.num_source_rows]
    result = 3 * ldpc
    result += (m+ldpc) / 2 * hdpc
    result += 4.6 * parameters.num_coded_rows
    result *= 8
    return result

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
        result['encoding_multiplications'] = r10_encoding_complexity(parameters)
    elif code == 'RQ':
        result['encoding_multiplications'] = rq_encoding_complexity(parameters)
    result['encoding_multiplications'] *= parameters.num_columns

    # we decode each output vector separately
    if code == 'R10':
        result['decoding_multiplications'] = r10_decoding_complexity(parameters)
    elif code == 'RQ':
        result['decoding_multiplications'] = rq_decoding_complexity(parameters)
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
    assert np.allclose(pdf.sum(), 1), "sum of pdf should be 1, but is {}".format(pdf.sum())
    return pdf

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
        cachedir='./results/LT',
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

def main():
    # num_inputs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    # overhead_levels = np.linspace(0, 0.1)
    # plt.figure()
    # for K in num_inputs:
    #     plt.plot(
    #         overhead_levels,
    #         r10_pdf(overhead_levels, num_inputs=K),
    #         label="$K={}$".format(K),
    #     )
    # plt.title("R10 Completion PDF")
    # plt.xlabel("Relative Overhead")
    # plt.ylabel("Probability")
    # plt.xlim((0, 0.1))
    # plt.ylim((0, 1))
    # plt.legend()
    # plt.tight_layout()

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
    print(r10)

    plot.load_delay_plot(
        [heuristic,
         r10],
        [heuristic_plot_settings,
         r10_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=uncoded,
        show=False,
        xlim_bot=(2, 3000),
        ylim_top=(0.51, 0.58),
        ylim_bot=(0, 1600),
    )
    plt.savefig("./plots/180328/load_delay.png", dpi='figure')

    plot.encode_decode_plot(
        [heuristic,
         r10],
        [heuristic_plot_settings,
         r10_plot_settings],
        'num_partitions',
        xlabel=r'Partitions $T$',
        normalize=None,
        show=False,
        xlim_bot=(2, 3000),
        ylim_top=(0.2, 1),
        ylim_mid=(0, 0.00035),
        ylim_bot=(0, 0.8),
    )
    plt.savefig("./plots/180328/phases.png", dpi='figure')
    plt.show()

if __name__ == '__main__':
    main()
