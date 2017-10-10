
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model
import evaluation.overhead


def get_parameters_size():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = int(1e4)
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(rows_per_server=rows_per_server,
                                                                 rows_per_partition=rows_per_partition,
                                                                 min_num_servers=servers,
                                                                 code_rate=code_rate,
                                                                 muq=muq, num_columns=num_columns)
        parameters.append(par)
    return parameters

def get_parameters_partitioning():
    '''Get a list of parameters for the partitioning plot.'''
    rows_per_batch = 250
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/3
    num_partitions = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30,
                      40, 50, 60, 75, 100, 120, 125, 150, 200, 250,
                      300, 375, 500, 600, 750, 1000, 1500, 3000]

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(rows_per_batch=rows_per_batch, num_servers=num_servers, q=q,
                                     num_outputs=num_outputs, server_storage=server_storage,
                                     num_partitions=partitions)
        parameters.append(par)

    return parameters

def unique_rows_plot():
    parameters = get_parameters_size()[:5]
    plt.subplot('111')
    results = list()
    for p in parameters:
        rows = evaluation.overhead.rows_from_q(parameters=p)
        result = dict()
        result['rows'] = rows
        result['num_source_rows'] = p.num_source_rows
        results.append(result)
        print(result)

    df = pd.DataFrame(results)
    plt.plot(df['num_source_rows'], df['rows'] / df['num_source_rows'])
    plt.grid()
    plt.show()
    return

def main():
    overheads = np.linspace(1.2, 1.4, 10)
    parameters = get_parameters_size()[5:10]
    plt.subplot('111')
    for p in parameters:
        results = list()
        for overhead in overheads:
            df = evaluation.overhead.performance_from_overhead(
                parameters=p,
                overhead=overhead,
            )
            result = dict()
            result['overhead'] = overhead
            result['baseline'] = p.computational_delay()
            for label in df:
                result[label] = df[label].mean()
            results.append(result)
            print(result)

        df = pd.DataFrame(results)
        plt.plot(df['overhead'], df['delay'] / df['baseline'], label=p.num_servers)

    plt.grid()
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    unique_rows_plot()
