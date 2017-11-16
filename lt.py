import logging
import glob
import model
import pyrateless
import complexity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import scipy.integrate as integrate

from mpl_toolkits.mplot3d import Axes3D
from plot import get_parameters_size, get_parameters_size_2
from itertools import product, cycle
from functools import partial
from concurrent.futures import ProcessPoolExecutor

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
    if options and total < options['total']:
        options['total'] = total
        options['encoding'] = encoding_multiplications
        options['decoding'] = decoding_multiplications
    return total

def merge_cached(pattern=None, filename='merged.csv'):
    '''merge all df's that match pattern into a single df'''
    assert pattern is not None
    results = list()
    for df in [pd.read_csv(f) for f in glob.glob(pattern)]:
        try:
            result = {column:df[column].mean() for column in df}
            results.append(result)
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(results)
    df['avg_degree'] = _avg_degree_from_df(df)
    df['failure_prob_estimate'] = _failure_prob_from_df(df)
    df.to_csv(filename)
    return df

def _avg_degree_from_df(df):
    '''compute average degree for each row of df'''
    result = list()
    for row in df.itertuples():
        avg_degree = pyrateless.coding.stats.avg_degree_from_delta_mode(
            num_inputs=int(row.num_inputs),
            delta=row.delta,
            mode=row.mode,
        )
        result.append(avg_degree)
    return np.asarray(result)

def _failure_prob_from_df(df):
    '''compute average degree for each row of df'''
    result = list()
    for row in df.itertuples():
        soliton = pyrateless.Soliton(
            symbols=int(row.num_inputs),
            mode=int(row.mode),
            failure_prob=row.delta
        )
        try:
            failure_prob = pyrateless.optimize.analytic.decoding_failure_prob_estimate(
                soliton=soliton,
                num_inputs=int(row.num_inputs),
                overhead=row.target_overhead,
            )
        except ValueError:
            failure_prob = 1

        result.append(failure_prob)
    return np.asarray(result)

def plot_cached(filename='merged_400.csv', target_overhead=1.2, target_failure_probability=1e-2):
    '''plot the cached (c, delta) space using cached simulations'''
    df = pd.read_csv(filename)

    # filter by target failure probability
    # df = df.loc[np.isclose(df['target_failure_probability'], target_failure_probability), :].reset_index()

    # filter by target overhead
    df = df.loc[np.isclose(df['target_overhead'], target_overhead), :].reset_index()

    # filter by estimated failure probability
    df = df.loc[df['failure_prob_estimate'] <= target_failure_probability, :].reset_index()

    print(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('mode')
    ax.set_ylabel('delta')
    ax.set_zlabel('complexity')
    # ax.zaxis._set_scale('log')

    parameters = get_parameters_size_2()[1]
    dct = heuristic_parameters((parameters, target_overhead, target_failure_probability))

    df['encoding_multiplications'] *= parameters.num_columns
    df['decoding_multiplications'] *= parameters.num_outputs
    plt.plot(
        df['mode'],
        df['delta'],
        df['encoding_multiplications'] + df['decoding_multiplications'],
        '.',
    )
    plt.plot(
        [dct['mode']],
        [dct['delta']],
        [dct['encoding_multiplications'] + dct['decoding_multiplications']],
        'ro')

    plt.tight_layout()
    plt.grid()
    plt.show()
    return

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
        inactivation_density=0.1,
        strategy='empiric',
        options=callback,
    )
    callback.update({
        'c': result.x[0],
        'delta': result.x[1],
    })
    return callback

def heuristic_parameters(parameter_tuple):
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
    try:
        c, delta = pyrateless.heuristic(
        num_inputs=parameters.num_source_rows,
            target_failure_probability=target_failure_probability,
            target_overhead=target_overhead,
        )
    except ValueError:
        return {}

    mode = pyrateless.coding.stats.mode_from_delta_c(
        num_inputs=parameters.num_source_rows,
        delta=delta,
        c=c,
    )
    df = pyrateless.simulate({
        'num_inputs': parameters.num_source_rows,
        'failure_prob': delta,
        'mode': mode,
    }, overhead=target_overhead)
    result = {label:df[label].mean() for label in df}
    result['encoding_multiplications'] *= parameters.num_columns
    result['decoding_multiplications'] *= parameters.num_outputs
    callback['encoding'] = result['encoding_multiplications']
    callback['decoding'] = result['decoding_multiplications']
    callback['total'] = callback['encoding'] + callback['decoding']
    callback.update(result)
    callback.update({
        'c': c,
        'delta': delta,
    })
    return callback

def plots():
    df = pd.read_csv('heuristic_4.csv')
    title = 'Heuristic complexity, tfp='
    print(df)

    # compute bdc code performance
    bdc_load = list()
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

    # normalize the lt code performance by that of the bdc
    bdc_encode = np.asarray(bdc_encode)
    bdc_decode = np.asarray(bdc_decode)
    df['load'] /= bdc_load
    df['encoding'] /= bdc_encode
    df['decoding'] /= bdc_decode
    df['total'] /= bdc_encode + bdc_decode

    # plot load
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('input symbols')
    ax.set_ylabel('load, normalized')
    ax.set_title('Communication Load')
    for target_overhead in np.unique(df['target_overhead']):
        df1 = df.loc[np.isclose(df['target_overhead'], target_overhead), :].reset_index()
        color = next(colors)
        plt.semilogy(df1['source_rows'],
                     df1['load'],
                     color+'o-',
                     label='LT load, ' + str(target_overhead))

    ax.set_yscale('log')
    ax.set_yticks([1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.grid()
    plt.legend()
    plt.show()
    return

    # plot decoding complexity
    for tfp in np.unique(df['target_failure_probability']):
        df1 = df.loc[np.isclose(df['target_failure_probability'], tfp), :].reset_index()

        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('input symbols')
        ax.set_ylabel('multiplications, normalized')
        ax.set_title(title + str(tfp))
        # plt.ylim(0.1, 10)
        for target_overhead in np.unique(df1['target_overhead']):
            df2 = df1.loc[np.isclose(df1['target_overhead'], target_overhead), :].reset_index()

            color = next(colors)
            plt.semilogy(df2['source_rows'],
                         df2['encoding'],
                         color+'d-',
                         label='encoding, ' + str(target_overhead))
            plt.semilogy(df2['source_rows'],
                         df2['decoding'],
                         color+'o-',
                         label='decoding, ' + str(target_overhead))
            plt.semilogy(df2['source_rows'],
                         df2['total'],
                         color+'x-',
                         label='enc+dec, ' + str(target_overhead))

        ax.set_yscale('log')
        ax.set_yticks([10, 5, 4, 3, 2, 1, 0.8, 0.6])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.grid()
        plt.legend()
        # plt.savefig(title + str(tfp) + '.png')

    plt.show()

def plot_tfp(target_failure_probability=1e-2, target_overhead=1.1):
    parameters = get_parameters_size()[0]
    # c, delta = pyrateless.heuristic(
    #     num_inputs=parameters.num_source_rows,
    #     target_failure_probability=target_failure_probability,
    #     target_overhead=target_overhead,
    # )

    # # compute the robust Soliton distribution mode
    # mode = pyrateless.coding.stats.mode_from_delta_c(
    #     num_inputs=parameters.num_source_rows,
    #     delta=delta,
    #     c=c,
    # )


    c, delta, mode = (0.00158028992469, 0.18017545342445374, 3999.0)

    # create a distribution object
    soliton = pyrateless.Soliton(
        symbols=parameters.num_source_rows,
        mode=mode,
        failure_prob=delta,
    )

    overhead = np.linspace(1, 2, 100)
    data = [1-pyrateless.optimize.decoding_failure_prob_estimate(
        soliton=soliton,
        num_inputs=parameters.num_source_rows,
        overhead=x,
    ) for x in overhead]
    integral = integrate.quad(
        lambda x: 1-pyrateless.optimize.decoding_failure_prob_estimate(
            soliton=soliton,
            num_inputs=parameters.num_source_rows,
            overhead=x,
        ), 1, 20
    )
    print('DATA:', data)
    print('integral:', integral)

    plt.figure()
    plt.plot(overhead, data)
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
    # for parameter_tuple in parameter_tuples:
    #     result = heuristic_parameters(parameter_tuple)
    #     if not result:
    #         continue
    #     results.append(result)
    #     df = pd.DataFrame(results)
    #     df.to_csv('heuristic_4.csv', index=False)
    #     print(result)

    with ProcessPoolExecutor() as executor:
        for result in executor.map(heuristic_parameters, parameter_tuples):
            if not result:
                continue
            results.append(result)
            df = pd.DataFrame(results)
            df.to_csv('heuristic_4.csv', index=False)
            print(result)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv('heuristic_4.csv', index=False)
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # for p in get_parameters_size():
    #     print(p)
    # print('--------------------')
    
    # main()
    # plots()
    # num_inputs = 3400
    # pattern = './pyrateless_cache_1/' + str(num_inputs) + '*.csv'
    # merge_cached(pattern=pattern, filename='merged_3400.csv')
    # plot_cached()
    plot_tfp()
