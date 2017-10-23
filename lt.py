import logging
import model
import pyrateless
import complexity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product, cycle
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def get_parameters_size_2():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 200
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = int(1e4)
    parameters = list()
    # num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    num_servers = [2, 5, 8, 20, 50] # , 200]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(rows_per_server=rows_per_server,
                                                                 rows_per_partition=rows_per_partition,
                                                                 min_num_servers=servers,
                                                                 code_rate=code_rate,
                                                                 muq=muq, num_columns=num_columns)
        parameters.append(par)
    return parameters

def _objective_function(
        encoding_additions=None,
        encoding_multiplications=None,
        decoding_additions=None,
        decoding_multiplications=None,
        options=None,
        parameters=None):
    '''sample objective function for minimizing the overall complexity of encoding and decoding'''
    encoding_multiplications *= parameters.num_columns # number of vectors to encode
    decoding_multiplications *= parameters.num_outputs # number of vectors to decode
    total = encoding_multiplications + decoding_multiplications
    if total < options['total']:
        options['total'] = total
        options['encoding'] = encoding_multiplications
        options['decoding'] = decoding_multiplications
    return total

def optimize_parameters(parameter_tuple):
    parameters = parameter_tuple[0]
    target_overhead = parameter_tuple[1]
    target_failure_probability = parameter_tuple[2]
    f = partial(
        _objective_function,
        parameters=parameters,
    )
    callback = parameters.asdict()
    callback.update({
        'encoding': 0,
        'decoding': 0,
        'total': float('inf'),
        'target_overhead': target_overhead,
        'target_failure_probability': target_failure_probability,
        'load': parameters.unpartitioned_load(overhead=target_overhead),
    })
    result = pyrateless.minimize(
        objective_function=f,
        num_inputs=parameters.num_source_rows,
        target_failure_probability=target_failure_probability,
        target_overhead=target_overhead,
        rate=parameters.q / parameters.num_servers,
        num_columns=1,
        inactivation_density=0.2,
        strategy='empiric',
        options=callback,
    )
    return callback

def plots():
    df = pd.read_csv('empiric_1.csv')
    bdc_load = list()
    rs_encode = list()
    rs_decode = list()
    bdc_encode = list()
    bdc_decode = list()
    bdc_num_inputs = list()
    for dct in df.to_dict('records'):
        parameters = model.SystemParameters.fromdct(dct)
        bdc_load.append(parameters.unpartitioned_load())
        bdc_num_inputs.append(parameters.num_source_rows)
        bdc_decode.append(complexity.block_diagonal_decoding_complexity(
            parameters.num_coded_rows,
            1,
            1 - parameters.q / parameters.num_servers,
            parameters.num_partitions
        ) * parameters.num_outputs)
        bdc_encode.append(
            complexity.block_diagonal_encoding_complexity(parameters)
        )

    df['load'] /= bdc_load
    df['encoding'] /= bdc_encode
    df['decoding'] /= bdc_decode
    df['total'] = df['encoding'] + df['decoding']

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('input symbols')
    ax.set_ylabel('load')
    for target_overhead in np.unique(df['target_overhead']):
        df1 = df.loc[np.isclose(df['target_overhead'], target_overhead), :].reset_index()
        color = next(colors)
        plt.plot(df1['source_rows'],
                 df1['load'],
                 color+'d-',
                 label='LT load, ' + str(target_overhead))
    plt.grid()
    plt.legend()
    plt.show()

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('input symbols')
    ax.set_ylabel('complexity')
    for target_overhead in np.unique(df['target_overhead']):
        df1 = df.loc[np.isclose(df['target_overhead'], target_overhead), :].reset_index()
        for target_failure_probability in np.unique(df1['target_failure_probability']):
            df2 = df1.loc[np.isclose(df1['target_failure_probability'],
                                     target_failure_probability), :].reset_index()
            color = next(colors)
            plt.semilogy(df2['source_rows'],
                         df2['encoding'],
                         color+'d-',
                         label='encoding, ' + str(target_overhead) + ', ' + str(target_failure_probability))
            plt.semilogy(df2['source_rows'],
                         df2['decoding'],
                         color+'o-',
                         label='decoding, ' + str(target_overhead) + ', ' + str(target_failure_probability))
            plt.semilogy(df2['source_rows'],
                         df2['total'],
                         color+'x-',
                         label='enc+dec, ' + str(target_overhead) + ', ' + str(target_failure_probability))

    plt.grid()
    plt.legend()
    plt.show()

def main():
    parameter_iter = get_parameters_size_2()
    target_overhead_iter = [1.1, 1.2, 1.3]
    target_failure_probability_iter = [1e-2, 1e-3, 1e-5]
    parameter_tuples = product(
        parameter_iter,
        target_overhead_iter,
        target_failure_probability_iter
    )
    results = list()
    with ProcessPoolExecutor() as executor:
        for result in executor.map(optimize_parameters, parameter_tuples):
            results.append(result)
            print(result)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('empiric_2.csv', index=False)
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # main()
    for p in get_parameters_size_2():
        print(p)
    # plots()
