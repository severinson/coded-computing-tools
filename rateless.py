'''Optimize LT codes for coded computing

'''

import logging
import numpy as np
import complexity
import stats

from functools import partial
from model import SystemParameters
from pyrateless import minimize

def _encoding_delay_operations(parameters=None, operations=None):
    '''compute encoding delay when it requires operations operations'''
    assert isinstance(parameters, SystemParameters)
    assert operations > 0
    delay = stats.order_mean_shiftexp(parameters.num_servers, parameters.num_servers)

    # Scale by encoding complexity
    delay *= operations / parameters.num_servers

    # the number of operations give is per column
    delay *= parameters.num_columns

    return delay

def _decoding_delay_operations(parameters=None, operations=None):
    '''compute decoding delay when it requires operations operations'''
    assert isinstance(parameters, SystemParameters)
    assert operations > 0
    delay = stats.order_mean_shiftexp(parameters.q, parameters.q)

    # Scale by encoding complexity
    delay *= operations

    # scale by number of output vectors per server
    delay *= parameters.num_outputs / parameters.q
    return delay

def _objective_function(
        encoding_additions=None,
        encoding_multiplications=None,
        decoding_additions=None,
        decoding_multiplications=None,
        overhead=None,
        max_overhead=1.3,
        parameters=None):
    '''optimization objective function'''
    assert isinstance(parameters, SystemParameters)
    if overhead > max_overhead:
        return np.finfo(float).max / 2
    delay = _encoding_delay_operations(
        parameters=parameters,
        operations=encoding_multiplications,
    )
    delay += _decoding_delay_operations(
        parameters=parameters,
        operations=decoding_multiplications,
    )
    delay += parameters.computational_delay()

    # bdc_delay = complexity.partitioned_encode_delay(parameters)
    # bdc_delay += complexity.partitioned_reduce_delay(parameters)
    # bdc_delay += parameters.computational_delay()
    # print(delay, bdc_delay, delay/bdc_delay)

    return delay

def optimize_parameters(parameters, max_overhead=1.3):
    '''optimize LT code for some parameters

    args:

    parameters: system parameters to optimize the LT code for.

    returns: total number of multiplications required for the encoding and
    decoding. the optimization is carried out such that decoding is possible
    with any q servers.

    returns: list with the combined delay due to encoding and decoding. the map
    phase delay must be added separately.

    '''
    failure_prob = 1e-2
    # rates = [parameters[0].q / parameters[0].num_servers]

    # TODO: Remove
    # num_inputs = [p.num_source_rows / 10 for p in parameters]
    # results = batch_minimize()
    delays = list()
    for p in parameters:
        num_inputs = p.num_source_rows
        rate = p.q / p.num_servers

        # optimize lt code parameters
        result = minimize(
            objective_function=partial(
                _objective_function,
                parameters=p,
                max_overhead=max_overhead,
            ),
            num_inputs=num_inputs,
            failure_prob=failure_prob,
            rate=rate
        )

        # the objective function value is the overall delay
        if result.success and result.fun < np.finfo(float).max / 4:
            lt_delay = result.fun
        else:
            lt_delay = 0

        # store bdc and rs code delays for comparison
        bdc_delay = complexity.partitioned_encode_delay(p)
        bdc_delay += complexity.partitioned_reduce_delay(
            p,
            erasure_probability=1-rate*max_overhead)
        bdc_delay += p.computational_delay()

        rs_delay = complexity.partitioned_encode_delay(p, partitions=1)
        rs_delay += complexity.partitioned_reduce_delay(
            p,
            partitions=1,
            erasure_probability=1-rate*max_overhead)
        rs_delay += p.computational_delay()

        delays.append({
            'lt': lt_delay,
            'bdc': bdc_delay,
            'rs': rs_delay,
        })
    return delays

def main():
    num_inputs = 100
    failure_prob = 1e-2
    rate = 2/3
    result = minimize(
        objective_function=min_complexity_objective_function,
        num_inputs=num_inputs,
        failure_prob=failure_prob,
        rate=rate
    )
    print(result)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
