import matplotlib.pyplot as plt
import model
import simulation
import complexity
import rateless
import ita_plots as ita
import plot
import stats

from evaluation.binsearch import SampleEvaluator
from evaluation import analytic
from solvers.heuristicsolver import HeuristicSolver
from functools import partial

def get_parameters_size():
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

def rateless_evaluate(parameters, pdf_fun=None):
    '''evaluate LT code performance.

    args:

    parameters: system parameters.

    returns: dict with performance results.

    '''
    assert isinstance(parameters, model.SystemParameters)
    result = dict()

    # we encode each column of the input matrix separately
    result['encoding_multiplications'] = r10_encoding_complexity(parameters)
    result['encoding_multiplications'] *= parameters.num_columns

    # we decode each output vector separately
    result['decoding_multiplications'] = r10_decoding_complexity(parameters)
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
        target_overhead=1+20/parameters.num_source_rows,
        mode=0,
        delta=0,
        pdf_fun=pdf_fun,
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
        pdf_fun=rateless.random_fountain_success_pdf,
    ),
)

r10_plot_settings = {
    'label': r'R10',
    'color': 'g',
    'marker': 's-',
    'linewidth': 4,
    'size': 2}
lt_plot_settings = {
    'label': r'LT',
    'color': 'c',
    'marker': 'v-',
    'linewidth': 4,
    'size': 2}

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

    result = 3 * num_ldpc[parameters.num_source_rows]
    result += parameters.num_source_rows / 2 * num_hdpc[parameters.num_source_rows]
    result += 4.6 * parameters.num_coded_rows
    result *= 8
    return result

def r10_decoding_complexity(parameters):
    '''assuming 25 XORs per source symbol'''
    assert isinstance(parameters, model.SystemParameters)
    return 25 * parameters.num_source_rows

def size_plot():
    parameters = get_parameters_size()

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

    plot.load_delay_plot(
        [heuristic,
         lt,
         r10],
        [ita.heuristic_plot_settings,
         lt_plot_settings,
         r10_plot_settings],
        'num_servers',
        xlabel=r'Servers $K$',
        normalize=uncoded,
        show=False,
    )
    plt.show()
    return

if __name__ == '__main__':
    size_plot()
