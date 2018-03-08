import logging
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

def rateless_evaluate(parameters, code='R10', pdf_fun=None):
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
        code='R10',
        pdf_fun=rateless.random_fountain_success_pdf,
    ),
)

rq_fun = partial(
    simulation.simulate,
    directory='./results/RQ/',
    samples=1,
    parameter_eval=partial(
        rateless_evaluate,
        code='RQ',
        pdf_fun=rateless.random_fountain_success_pdf,
    ),
)

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

def rq_encoding_complexity(parameters):
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
        4000: 11,
        6000: 11,
        14000: 12,
        34000: 14,
        54000: 16,
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
    result += 2 * (m+ldpc) * hdpc
    result += 4.82 * parameters.num_coded_rows
    result *= 8
    return result

def r10_decoding_complexity(parameters):
    '''assuming 25 XORs per source symbol'''
    assert isinstance(parameters, model.SystemParameters)
    return 25 * parameters.num_source_rows

def rq_decoding_complexity(parameters):
    return r10_decoding_complexity()

def size_plot():
    parameters = get_parameters_size_20()

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
    parameters = get_parameters_size_10()
    parameters[-3:] = get_parameters_size_20()[-3:]
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
        [ita.heuristic_plot_settings,
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
    plt.savefig("./plots/180309/load_delay.png")

    plot.encode_decode_plot(
        [heuristic,
         lt,
         r10,
         rq],
        [ita.heuristic_plot_settings,
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
    plt.savefig("./plots/180309/encode_decode.png")

    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    size_plot()
