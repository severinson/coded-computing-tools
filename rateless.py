'''Optimize rateless codes for distributed computing

'''

import logging
import numpy as np
import pandas as pd
import pyrateless
import stats
import complexity
import overhead

from functools import lru_cache
from plot import get_parameters_size_2

def simulate_parameter_list(parameter_list):
    '''evaluate a list of parameters

    returns: dataframe with the result

    '''
    results = list()
    for parameters in parameter_list:
        results.append(evaluate(parameters))

    return pd.DataFrame(results)

def evaluate(parameters, target_overhead=None, target_failure_probability=None):
    '''evaluate LT code performance.

    args:

    parameters: system parameters.

    returns: dict with performance results.

    '''

    # find good LT code parameters
    c, delta = pyrateless.heuristic(
        num_inputs=parameters.num_source_rows,
        target_failure_probability=target_failure_probability,
        target_overhead=target_overhead,
    )

    # compute the robust Soliton distribution mode
    mode = pyrateless.coding.stats.mode_from_delta_c(
        num_inputs=parameters.num_source_rows,
        delta=delta,
        c=c,
    )

    logging.debug(
        'LT mode=%d, delta=%f for %d input symbols, target overhead %f, target failure probability %f',
        mode, delta, parameters.num_source_rows, target_overhead, target_failure_probability,
    )

    # simulate the the code performance
    df = pyrateless.simulate({
        'num_inputs': parameters.num_source_rows,
        'failure_prob': delta,
        'mode': mode,
    }, overhead=target_overhead)

    # average the columns of the df
    mean = {label:df[label].mean() for label in df}

    # store the results we return in a new dict. we do not return the mean dict
    # as we do not scale all of its values correctly when the number of columns
    # or outputs is not 1.
    result = dict()

    # we encode each column of the input matrix separately
    result['encoding_multiplications'] = mean['encoding_multiplications']
    result['encoding_multiplications'] *= parameters.num_columns /parameters.num_servers

    # we decode each output vector separately
    result['decoding_multiplications'] = mean['decoding_multiplications']
    result['decoding_multiplications'] *= parameters.num_outputs / parameters.q

    # compute encoding delay
    result['encode'] = stats.order_mean_shiftexp(
        parameters.num_servers,
        parameters.num_servers,
        parameter=result['encoding_multiplications'],
    )

    # compute decoding delay
    result['reduce'] = stats.order_mean_shiftexp(
        parameters.q,
        parameters.q,
        parameter=result['decoding_multiplications'],
    )

    # compute the complexity of the map computation (the matrix multiplication)
    rows_per_server = parameters.server_storage * parameters.num_source_rows
    map_complexity = complexity.matrix_vector_complexity(
        rows=rows_per_server,
        cols=parameters.num_columns,
    )

    # simulate the map phase load/delay
    simulated = performance_integral(
        parameters=parameters,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
    )
    result['delay'] = simulated['delay']
    result['load'] = simulated['load']

    # scale the map delay by its complexity
    result['delay'] *= map_complexity

    # add encoding/decoding to the overall delay
    result['delay'] += result['encode']
    result['delay'] += result['reduce']

    # normalize
    result['delay'] /= parameters.num_source_rows
    result['encode'] /= parameters.num_source_rows
    result['reduce'] /= parameters.num_source_rows

    # store some parameters to plot against
    result['servers'] = parameters.num_servers
    result['partitions'] = parameters.num_partitions
    result['num_inputs'] = parameters.num_outputs
    result['num_columns'] = parameters.num_columns

    return result

@lru_cache()
def performance_integral(parameters=None, target_overhead=None,
                         mode=None, delta=None, samples=100):
    '''compute average performance by taking into account the probability of
    finishing at different levels of overhead.

    '''

    # get the max possible overhead
    max_overhead = parameters.num_coded_rows / parameters.num_source_rows

    # evaluate the performance at various levels of overhead
    overhead_levels = np.linspace(target_overhead, max_overhead, samples)

    # create a distribution object
    soliton = pyrateless.Soliton(
        symbols=parameters.num_source_rows,
        mode=mode,
        failure_prob=delta,
    )

    # compute the probability of decoding at discrete levels of overhead. the
    # first element is zero to take make the pdf sum to 1.
    decoding_cdf = np.fromiter(
        [0] + [1-pyrateless.optimize.decoding_failure_prob_estimate(
            soliton=soliton,
            num_inputs=parameters.num_source_rows,
            overhead=x) for x in overhead_levels
        ], dtype=float)
    decoding_pdf = np.diff(decoding_cdf)

    # print('cdf', decoding_cdf, len(decoding_cdf))
    # print('pdf', decoding_pdf, len(decoding_pdf))
    # print('overhead', overhead_levels, len(overhead_levels))

    # compute load/delay at the levels of overhead
    results = list()
    for overhead_level, decoding_probability in zip(overhead_levels, decoding_pdf):

        # monte carlo simulation of the load/delay at this overhead
        df = overhead.performance_from_overhead(
            parameters=parameters,
            overhead=overhead_level,
        )

        # average the columns of the df
        result = {label:df[label].mean() for label in df}

        # multiply by the probability of decoding at this overhead level
        for label in result:
            result[label] *= decoding_probability

        results.append(result)

    # create a dataframe and sum along the columns
    df = pd.DataFrame(results)
    return {label:df[label].sum() for label in df}
