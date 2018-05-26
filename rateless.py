'''Optimize rateless codes for distributed computing

'''

import math
import random
import logging
import numpy as np
import pandas as pd
import pyrateless
import stats
import complexity
import overhead
import pynumeric
import tempfile
import subprocess

from os import path
from multiprocessing import Pool

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

def lt_encoding_complexity(num_inputs=None, failure_prob=None,
                           target_overhead=None, code_rate=None):
    '''Return the decoding complexity of LT codes. Computed from the
    average of the degree distribution.

    The number of columns is assumed to be 1. Scale the return value
    of this function by the actual number of columns to get the
    correct complexity.

    '''
    # find good LT code parameters
    c, delta, mode = optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=failure_prob,
    )
    avg_degree = pyrateless.Soliton(
        delta=delta,
        mode=mode,
        symbols=num_inputs).mean()
    encoding_complexity = pyrateless.optimize.complexity.encoding_additions(
        avg_degree,
        code_rate,
        num_inputs,
        1, # number of columns
    ) * complexity.ADDITION_COMPLEXITY
    encoding_complexity += pyrateless.optimize.complexity.encoding_multiplications(
        avg_degree,
        code_rate,
        num_inputs,
        1,
    ) * complexity.MULTIPLICATION_COMPLEXITY
    return encoding_complexity

def lt_decoding_complexity(num_inputs=None, failure_prob=None,
                           target_overhead=None):
    '''Return the decoding complexity of LT codes. Data is manually
    entered from simulations carried out using
    https://github.com/severinson/RaptorCodes

    '''

    # maps a tuple (num_inputs, target_failure_probability,
    # target_overhead) to a tuple (num_inactivations,
    # num_row_operations). contains simulated results.
    if failure_prob == 1e-1:
        filename = './results/LT_1e-1.csv'
    elif failure_prob == 1e-3:
        filename = './results/LT_1e-3.csv'

    try:
        df = pd.read_csv(filename)
    except:
        logging.error('could not load file {}.'.format(filename))
        return math.inf

    overhead = round(num_inputs*(target_overhead-1))
    df = df.loc[df['num_inputs'] == num_inputs]
    df = df.loc[df['overhead'] == overhead]
    if len(df) != 1:
        logging.warning(
            'did not find exactly 1 row for num_inputs={}, failure_prob={}, target_overhead=: {} symbols'.format(
                num_inputs, failure_prob, overhead, df,))
        return math.inf

    a = df['diagonalize_decoding_additions']
    a += df['diagonalize_rowadds']
    a += df['solve_dense_decoding_additions']
    a += df['solve_dense_rowadds']
    a += df['backsolve_decoding_additions']
    a += df['backsolve_rowadds']
    a = a.values[0]
    m = df['diagonalize_decoding_multiplications']
    m += df['diagonalize_rowmuls']
    m += df['solve_dense_decoding_multiplications']
    m += df['solve_dense_rowmuls']
    m += df['backsolve_decoding_multiplications']
    m += df['backsolve_rowmuls']
    m = m.values[0]
    return a*complexity.ADDITION_COMPLEXITY + m*complexity.MULTIPLICATION_COMPLEXITY

def evaluate(parameters, target_overhead=None,
             target_failure_probability=None,
             pdf_fun=None, partitioned=False,
             cachedir=None):
    '''evaluate LT code performance.

    args:

    parameters: system parameters.

    pdf_fun: see rateless.performance_integral

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

    # compute encoding complexity
    encoding_complexity = lt_encoding_complexity(
        num_inputs=num_inputs,
        failure_prob=target_failure_probability,
        target_overhead=target_overhead,
        code_rate=parameters.q/parameters.num_servers,
    )
    encoding_complexity *= parameters.num_columns
    encoding_complexity *= num_partitions
    encoding_complexity *= parameters.muq

    # compute decoding complexity
    decoding_complexity = lt_decoding_complexity(
        num_inputs=num_inputs,
        failure_prob=target_failure_probability,
        target_overhead=target_overhead,
    )
    decoding_complexity *= num_partitions
    decoding_complexity *= parameters.num_outputs

    # find good code parameters
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

    # scale the number of multiplications required for encoding/decoding and
    # store in a new dict.
    result = dict()

    # compute encoding delay
    result['encode'] = stats.order_mean_shiftexp(
        parameters.num_servers,
        parameters.num_servers,
        parameter=encoding_complexity / parameters.num_servers,
    )

    # compute decoding delay
    result['reduce'] = stats.order_mean_shiftexp(
        parameters.q,
        parameters.q,
        parameter=decoding_complexity / parameters.q,
    )

    # simulate the map phase load/delay. this simulation takes into account the
    # probability of decoding at various levels of overhead.
    simulated = performance_integral(
        parameters=parameters,
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
        pdf_fun=pdf_fun,
        cachedir=cachedir,
    )
    result['delay'] = simulated['delay']
    result['load'] = simulated['load']
    return result

def lt_success_pdf(overhead_levels, num_inputs=None, mode=None, delta=None):
    '''evaluate the decoding probability pdf.

    args:

    overhead_levels: iterable of overhead levels to evaluate the PDF at.

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
    return decoding_pdf / decoding_pdf.sum()

def lt_success_samples(n, target_overhead=None, num_inputs=None, mode=None, delta=None):
    '''sample the decoding probability distribution.

    '''
    assert n > 0
    assert n % 1 == 0
    if target_overhead is None:
        target_overhead = 1
    # create a distribution object. this is needed for the decoding success
    # probability estimate.
    soliton = pyrateless.Soliton(
        symbols=num_inputs,
        mode=mode,
        failure_prob=delta,
    )
    cdf = lambda x: 1-pyrateless.optimize.decoding_failure_prob_estimate(
        soliton=soliton,
        num_inputs=num_inputs,
        overhead=x,
    )

    # with Pool(processes=12) as pool:
    samples = np.fromiter((
        pynumeric.cnuminv(
            fun=cdf,
            target=random.random(),
            lower=target_overhead,
        ) for _ in range(n)), dtype=float)
    return np.maximum(samples, target_overhead)

def random_fountain_success_pdf(overhead_levels, field_size=2, num_inputs=None, mode=None, delta=None):
    '''compute the decoding success probability PDF of a random fountain code over
    a field of size field_size.

    '''
    assert field_size % 1 == 0, field_size
    absolute_overhead = np.fromiter(
        (num_inputs*(x-1) for x in overhead_levels),
        dtype=float,
    ).round()
    if absolute_overhead.min() < 0:
        raise ValueError("error for overhead levels {}. overhead must be >=1.".format(overhead_levels))
    decoding_cdf = 1-np.power(field_size, -absolute_overhead)
    decoding_pdf = np.zeros(len(decoding_cdf))
    decoding_pdf[1:] = np.diff(decoding_cdf)
    decoding_pdf[0] = decoding_cdf[0]
    return decoding_pdf

def performance_integral(parameters=None, num_inputs=None, target_overhead=None,
                         mode=None, delta=None, pdf_fun=None, num_overhead_levels=100,
                         max_overhead=None, cachedir=None):
    '''compute average performance by taking into account the probability of
    finishing at different levels of overhead.

    pdf_fun: function used to evaluate the decoding success probability.
    defaults to rateless.lt_success_pdf if None. a function given here
    must have the same signature as this function.

    num_overhead_levels: performance is evaluated at num_overhead_levels levels
    of overhead between target_overhead and the maximum possible overhead.

    '''
    if pdf_fun is None:
        pdf_fun = lt_success_pdf
    assert callable(pdf_fun)

    # get the max possible overhead
    if max_overhead is None:
        max_overhead = parameters.num_coded_rows / parameters.num_source_rows

    if max_overhead < target_overhead:
        raise ValueError("target overhead may not exceed the inverse of the code rate")

    # evaluate the performance at various levels of overhead
    overhead_levels = np.linspace(target_overhead, max_overhead, num_overhead_levels)

    # compute the probability of decoding at the respective levels of overhead
    decoding_probabilities = pdf_fun(
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
            cachedir=cachedir,
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
              partitioned=False, num_overhead_levels=100, num_samples=100000,
              cachedir=None):
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
    decoding_probabilities = lt_success_pdf(
        overhead_levels,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )

    # simulate the number of servers needed at each level of overhead. the
    # number of samples taken is weighted by the probability of needing this
    # overhead.
    results = list()
    print(overhead_levels)
    print(decoding_probabilities)
    print(decoding_probabilities.sum())
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
    order_count.sort_index(inplace=True)
    order_values = np.array(order_count.index)
    order_probabilities = order_count.values
    return order_values, order_probabilities
