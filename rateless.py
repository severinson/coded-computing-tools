'''Optimize rateless codes for distributed computing

'''

import logging
import numpy as np
import pandas as pd
import pyrateless
import stats
import complexity
import overhead

def optimize_lt_parameters(num_inputs=None, target_overhead=None,
                           target_failure_probability=None):
    '''find good lt code parameters

    returns: a tuple (c, delta, mode)

    '''
    c, delta = pyrateless.heuristic(
        num_inputs=num_inputs,
        target_failure_probability=target_failure_probability,
        target_overhead=target_overhead,
    )

    # compute the robust Soliton distribution mode
    mode = pyrateless.coding.stats.mode_from_delta_c(
        num_inputs=num_inputs,
        delta=delta,
        c=c,
    )
    return c, delta, mode

def evaluate(parameters, target_overhead=None,
             target_failure_probability=None, partitioned=False):
    '''evaluate LT code performance.

    args:

    parameters: system parameters.

    partitioned: evaluate the performance of the scheme using a partitioned LT
    code with rows_per_batch number of partitions. this case is easy to
    evaluate as we will always receive the same coded symbols for each
    partition. in particular, if it is possible to decode one partition, we can
    decode all others as well. this is only true for
    num_partitions=rows_per_batch.

    returns: dict with performance results.

    '''
    assert target_overhead > 1
    assert 0 < target_failure_probability < 1
    assert isinstance(partitioned, bool)

    # we support only either no partitioning or exactly rows_per_batch
    # partitions. this case is much simpler to handle due to all partitions
    # behaving the same only in this instance.
    if partitioned:
        num_partitions = parameters.rows_per_batch
    else:
        num_partitions = 1

    # guaranteed to be an integer
    num_inputs = int(parameters.num_source_rows / num_partitions)

    # find good LT code parameters
    c, delta, mode = optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )

    logging.debug(
        'LT mode=%d, delta=%f for %d input symbols, target overhead %f, target failure probability %f. partitioned: %r',
        mode, delta, parameters.num_source_rows,
        target_overhead, target_failure_probability,
        partitioned,
    )

    # simulate the the code performance. we only extract the number of
    # multiplications required for encoding and decoding from this simulation.
    df = pyrateless.simulate({
        'num_inputs': num_inputs,
        'failure_prob': delta,
        'mode': mode,
    }, overhead=target_overhead)

    # average the columns of the df
    mean = {label:df[label].mean() for label in df}

    # scale the number of multiplications required for encoding/decoding and
    # store in a new dict.
    result = dict()

    # we encode each column of the input matrix separately
    result['encoding_multiplications'] = mean['encoding_multiplications']
    result['encoding_multiplications'] *= parameters.num_columns

    # we decode each output vector separately
    result['decoding_multiplications'] = mean['decoding_multiplications']
    result['decoding_multiplications'] *= parameters.num_outputs

    # scale by the number of partitions
    result['encoding_multiplications'] *= num_partitions
    result['decoding_multiplications'] *= num_partitions

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
    simulated = performance_integral(
        parameters=parameters,
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
    )
    result['delay'] = simulated['delay']
    result['load'] = simulated['load']
    return result

def decoding_success_pdf(overhead_levels, num_inputs=None, mode=None, delta=None):
    '''evaluate the decoding probability pdf.

    args:

    overhead_levels: levels of overhead to evaluate the PDF at.

    num_inputs: number of input symbols.

    returns: a vector of the same length as overhead_levels, where i-th element
    is the probability of decoding at an overhead of overhead_levels[i].

    '''

    # create a distribution object. this is needed for the decoding success
    # probability estimate.
    soliton = pyrateless.Soliton(
        symbols=num_inputs,
        mode=mode,
        failure_prob=delta,
    )

    # compute the probability of decoding at discrete levels of overhead. the
    # first element is zero to take make the pdf sum to 1.
    decoding_cdf = np.fromiter(
        [0] + [1-pyrateless.optimize.decoding_failure_prob_estimate(
            soliton=soliton,
            num_inputs=num_inputs,
            overhead=x) for x in overhead_levels
        ], dtype=float)

    # differentiate the CDF to obtain the PDF
    decoding_pdf = np.diff(decoding_cdf)
    return decoding_pdf

def performance_integral(parameters=None, num_inputs=None, target_overhead=None,
                         mode=None, delta=None, num_overhead_levels=100):
    '''compute average performance by taking into account the probability of
    finishing at different levels of overhead.

    num_overhead_levels: performance is evaluated at num_overhead_levels levels
    of overhead between target_overhead and the maximum possible overhead.

    '''

    # get the max possible overhead
    max_overhead = parameters.num_coded_rows / parameters.num_source_rows

    # evaluate the performance at various levels of overhead
    overhead_levels = np.linspace(target_overhead, max_overhead, num_overhead_levels)

    # compute the probability of decoding at the respective levels of overhead
    decoding_probabilities = decoding_success_pdf(
        overhead_levels,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )

    # compute load/delay at the levels of overhead
    results = list()
    for overhead_level, decoding_probability in zip(overhead_levels, decoding_probabilities):

        # monte carlo simulation of the load/delay at this overhead
        df = overhead.performance_from_overhead(
            parameters=parameters,
            overhead=overhead_level,
            design_overhead=target_overhead,
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

def order_pdf(parameters=None, target_overhead=None, target_failure_probability=None,
              partitioned=False, num_overhead_levels=100, num_samples=100000):
    '''simulate the order PDF, i.e., the PDF over the number of servers needed to
    decode successfully.

    num_samples: total number of samples to take of the number of servers
    needed. the PDF is inferred from all samples.

    returns: two arrays (order_values, order_probabilities) with the possible
    number of servers needed and the probability of needing that number of
    servers, respectively.

    '''

    # we support only either no partitioning or exactly rows_per_batch
    # partitions. this case is much simpler to handle due to all partitions
    # behaving the same only in this instance.
    if partitioned:
        num_partitions = parameters.rows_per_batch
    else:
        num_partitions = 1

    # guaranteed to be an integer
    num_inputs = int(parameters.num_source_rows / num_partitions)

    # find good LT code parameters
    c, delta, mode = optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )

    # get the max possible overhead
    max_overhead = parameters.num_coded_rows / parameters.num_source_rows

    # evaluate the performance at various levels of overhead
    overhead_levels = np.linspace(target_overhead, max_overhead, num_overhead_levels)

    # compute the probability of decoding at the respective levels of overhead
    decoding_probabilities = decoding_success_pdf(
        overhead_levels,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )

    # simulate the number of servers needed at each level of overhead. the
    # number of samples taken is weighted by the probability of needing this
    # overhead.
    results = list()
    for overhead_level, decoding_probability in zip(overhead_levels, decoding_probabilities):

        # the number of samples correspond to the probability of decoding at
        # this level of overhead.
        overhead_samples = int(round(decoding_probability * num_samples))

        # monte carlo simulation of the load/delay at this overhead
        df = overhead.performance_from_overhead(
            parameters=parameters,
            overhead=overhead_level,
            design_overhead=target_overhead,
            num_samples=overhead_samples,
        )
        results.append(df)

    # concatenate all samples into a single dataframe
    samples = pd.concat(results, ignore_index=True)

    # compute the empiric order cdf and return
    order_count = samples['servers'].value_counts(normalize=True)
    order_values = np.array(order_count.index)
    order_probabilities = order_count.values
    return order_values, order_probabilities
