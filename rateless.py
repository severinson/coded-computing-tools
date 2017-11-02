'''Optimize rateless codes for distributed computing

'''

import pandas as pd
import pyrateless
import stats
import complexity

from plot import get_parameters_size_2

def simulate_parameter_list(parameter_list):
    '''evaluate a list of parameters

    returns: dataframe with the result

    '''
    results = list()
    for parameters in parameter_list:
        results.append(evaluate(parameters))

    return pd.DataFrame(results)

def evaluate(parameters, target_overhead=1.3, target_failure_probability=1e-2):
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

    # simulate the the code performance
    df = pyrateless.simulate({
        'num_inputs': parameters.num_source_rows,
        'failure_prob': delta,
        'mode': mode,
    }, overhead=target_overhead)

    # average the columns of the df
    result = {label:df[label].mean() for label in df}

    # scale the number of encoding/decoding multiplications
    result['encoding_multiplications'] *= parameters.num_columns
    result['decoding_multiplications'] *= parameters.num_outputs

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

    # compute map delay
    # TODO: We assume that q servers is always enough
    rows_per_server = parameters.server_storage * parameters.num_source_rows
    map_complexity = complexity.matrix_vector_complexity(
        rows=rows_per_server,
        cols=parameters.num_columns,
    ) * parameters.num_outputs
    result['delay'] = stats.order_mean_shiftexp(
        parameters.num_servers,
        parameters.q,
        parameter=map_complexity,
    )

    # delay should include everything
    result['delay'] += result['encode']
    result['delay'] += result['reduce']

    # normalize
    result['delay'] /= parameters.num_source_rows

    # load is computed from the target overhead
    result['load'] = parameters.unpartitioned_load(overhead=target_overhead)

    # store the number of servers and partitions to plot against later
    result['servers'] = parameters.num_servers
    result['partitions'] = parameters.num_partitions

    return result

