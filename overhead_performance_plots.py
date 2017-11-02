
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model
import overhead

from plot import get_parameters_size, get_parameters_size_2

def unique_rows_plot():
    parameters = get_parameters_size_2()
    plt.subplot('111')
    results = list()
    for p in parameters:
        rows = overhead.rows_from_q(parameters=p)
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
            df = overhead.performance_from_overhead(
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
