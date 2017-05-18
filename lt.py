'''Simulate Luby Transform code decoding.'''

import math
import random
import logging
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin
import numtools

class LTCode(object):
    '''Luby Transform code object. This object implements a peeling
    decoder.

    '''
    def __init__(self, symbols):
        '''Setup an LT code object.

        Args:

        symbols: Total number of uncoded symbols.

        '''
        self.symbols = symbols
        self.reset()
        return

    def reset(self):
        '''Reset the code.'''
        self.decoded = set()
        self.coded = dict()
        for i in range(self.symbols):
            self.coded[i] = list()
        self.stack = list()
        self.additions = 0
        self.coded_symbols = 0
        return

    def peel(self):
        '''Recursively peel coded symbols.'''
        i = 0

        # Start peeling.
        while self.stack and len(self.decoded) < self.symbols:
            i += 1

            # Pop a symbol index off the stack.
            index = self.stack.pop()

            # Add it to the set of decoded symbols.
            self.decoded.add(index)

            # Recursively peel any cached symbols including this one.
            for symbol in self.coded[index]:
                self.additions += len(symbol)
                symbol.discard(index)
                self.additions -= len(symbol)
                if len(symbol) == 1:
                    new_index = symbol.pop()
                    if new_index not in self.decoded:
                        self.stack.append(new_index)

            # Clear the symbols connected to this one.
            self.coded[index] = list()

        # logging.debug('Decoded %d symbols in one run.', i)
        return

    def add(self, symbol):
        '''Receive a coded symbol.

        Args:

        symbols: A set of symbol indices.

        '''
        self.coded_symbols += 1

        # Subtract any decoded symbols.
        self.additions += len(symbol)
        symbol = symbol - self.decoded
        self.additions -= len(symbol)

        # Start peeling if we decoded a new symbol.
        if len(symbol) == 1:
            # Add the index to the stack for processing.
            self.stack.append(symbol.pop())

            # Run the peeling process.
            self.peel()

        # Otherwise cache the symbol.
        else:
            for index in symbol:
                self.coded[index].append(symbol)

        return

    def completed(self):
        '''Return True if all symbols have been decoded.'''
        return len(self.decoded) == self.symbols

class Soliton(object):
    '''This object describes the robust Soliton distribution.'''

    def __init__(self, failure_prob, symbols, mode):
        '''Initialize the object.

        Args:

        failure_prob: Decoding failure probability.

        symbols: Total number of uncoded symbols.

        mode: Location of spike added by the robust component of the
        distribution.

        '''
        assert 0 < failure_prob < 1
        assert 0 < symbols < math.inf
        assert 0 < mode < symbols
        self.failure_prob = failure_prob
        self.symbols = symbols
        self.mode = np.round(mode)
        self.R = symbols / mode
        self.c = self.R / math.log(symbols / failure_prob) / math.sqrt(symbols)
        # self.R = c * math.log(symbols / failure_prob) * math.sqrt(symbols)
        self.beta = sum(self.tau(i) + self.rho(i) for i in range(1, symbols+1))
        return

    def __repr__(self):
        dct = {'failure_prob': self.failure_prob, 'symbols': self.symbols,
               'R': self.R, 'beta': self.beta, 'c': self.c}
        return str(dct)

    @functools.lru_cache(maxsize=2048)
    def tau(self, i):
        '''Added to the ideal distribution to produce the robust.'''
        assert 0 < i <= self.symbols
        if i < self.mode:
            return 1 / (i * self.mode)
        elif i == self.mode:
            return math.log(self.R / self.failure_prob) / self.mode
        else:
            return 0

    # def tau(self, i):
    #     '''Added to the ideal distribution to produce the robust.'''
    #     assert 0 < i <= self.symbols
    #     if i < self.symbols / self.R:
    #         return self.R / (i * self.symbols)
    #     elif i == self.symbols / self.R:
    #         return self.R * math.log(self.R / self.failure_prob) / self.symbols
    #     else:
    #         return 0

    @functools.lru_cache(maxsize=2048)
    def rho(self, i):
        '''The ideal Soliton distribution.'''
        assert 0 < i <= self.symbols
        if i == 1:
            return 1 / self.symbols
        else:
            return 1 / (i * (i - 1))

    @functools.lru_cache(maxsize=2048)
    def pdf(self, i):
        '''Robust Soliton distribution PDF.'''
        assert 0 < i <= self.symbols
        return (self.tau(i) + self.rho(i)) / self.beta

    @functools.lru_cache(maxsize=2048)
    def cdf(self, i):
        '''Robust Soliton distribution CDF.'''
        assert 0 < i <= self.symbols
        return sum(self.pdf(x) for x in range(1, i+1))

    def icdf(self, i):
        '''Robust Soliton distribution inverse CDF.'''
        return numtools.numerical_inverse(self.cdf, i, 1, self.symbols)

    @functools.lru_cache(maxsize=4)
    def mean(self):
        return sum(i * self.pdf(i) for i in range(1, self.symbols+1))

    def sample(self):
        '''Return a random sample.'''
        ivalue = random.random()
        return self.icdf(ivalue)

def lt_simulate(symbols, failprob=1/2, moderate=1/2, mode=None, filename='./ltsim.csv', eps=1, min_runs=10):
    '''Simulate the decoding performance of an LT code.

    Args:

    symbols: Total number of source symbols.

    failprob: Robust Soliton distribution delta parameter.

    moderate: Ratio between the robust Soliton distribution M
    parameter and the number of source symbols.

    mode: Robust Soliton distribution M parameter. Takes precendence
    over the moderate argument.

    filename: Simulation results are cached in this file.

    eps: Stop the simulation when it has converged within eps.

    min_runs: Minimum number of simulation runs to make.

    Returns: Pandas DataFrame with the simulated results.

    '''
    assert symbols > 0 and symbols % 1 == 0
    assert 0 < failprob < 1
    assert 0 < moderate <= symbols
    assert isinstance(filename, str)
    columns = ['symbols', 'failprob', 'mode', 'c', 'coded', 'additions']
    if mode is None:
        mode = round(symbols * moderate)

    try:
        dataframe = pd.read_csv(filename)
        results = dataframe[dataframe['symbols'] == symbols]
        results = results[results['failprob'] == failprob]
        results = results[results['mode'] == mode]
        if len(results) == 0:
            raise FileNotFoundError

        index = results['coded'].idxmin()
        return dict(dataframe.iloc[index])
    except FileNotFoundError:
        logging.info('Simulating LT decoding performance for %d symbols.', symbols)

    # Setup a Soliton random variable and a decoder.
    soliton = Soliton(failprob, symbols, mode)
    code = LTCode(symbols)

    # Simulate the decoding until convergence.
    runs = 0
    coded = math.inf
    total_coded = 0
    total_additions = 0
    while True:
        # Store the mean of last round.
        prev_coded = coded

        # Run a simulation.
        code.reset()
        while not code.completed():
            degree = soliton.sample()
            symbol = {random.randint(0, symbols-1) for _ in range(degree)}
            code.add(symbol)

        # Update the tally.
        total_coded += code.coded_symbols
        total_additions += code.additions
        runs += 1
        coded = total_coded / runs

        # Exit if the mean has converged.
        if runs > min_runs and abs(coded - prev_coded) < eps:
            break

    additions = total_additions / runs
    logging.debug('Converged to %d after %d runs. Mode: %d.',
                  coded, runs)

    result = {'symbols': symbols, 'failprob': failprob, 'mode': mode,
              'c': soliton.c, 'coded': coded, 'additions': additions}

    # Try loading from disk again. For if another process wrote
    # results in the meantime. Otherwise create a new dataframe.
    try:
        dataframe = pd.read_csv(filename)
        dataframe = dataframe.append(pd.DataFrame([result], columns=columns))
    except FileNotFoundError:
        dataframe = pd.DataFrame([result], columns=columns)

    # Save results to disk.
    dataframe.to_csv(filename, index=False)
    return result

def optimize_lt():
    '''Optimize LT code parameters.'''
    x0 = 500
    result = fmin(required_symbols, x0, xtol=1, ftol=1, full_output=True, disp=1)
    print(result)
    return

def soliton_pdf_plot():
    soliton = Soliton(0.1, 100, 5)
    x = np.arange(1, 101)
    y = [soliton.pdf(i) for i in x]
    print('Distribution sums to {}.'.format(sum(y)))
    plt.plot(x, y, '*')

    y = [soliton.cdf(i) for i in x]
    plt.plot(x, y)

    samples = [soliton.sample() for _ in range(1000)]
    plt.hist(samples, 100, normed=True, cumulative=True)
    plt.show()
    return

def required_symbols(mode, symbols=int(1e4), failure_prob=1/2, min_runs=10, eps=1):
    '''Simulate the required number of coded symbols.'''
    soliton = Soliton(failure_prob, symbols, mode)
    runs = 0
    total = 0
    mean = math.inf
    total_additions = 0
    while True:
        # Store the mean of last round.
        prev_mean = mean

        # Take a sample and update the mean.
        iterations, additions = required_sample(soliton, symbols)
        total += iterations
        total_additions += additions
        runs += 1
        mean = total / runs

        # Exit if the mean has converged.
        if runs > min_runs and abs(mean - prev_mean) < eps:
            break

    additions = total_additions / runs
    logging.info('Converged to %d after %d runs. Mode: %d. Additions: %d\nSoliton: %s.',
                 mean, runs, mode, additions, str(soliton))
    return mean, additions, soliton

def required_sample(soliton, symbols):
    '''Sample the required number of coded symbols.'''
    i = 0
    code = LTCode(symbols)
    while not code.completed():
        d = soliton.sample()
        symbol = {random.randint(0, symbols-1) for _ in range(d)}
        code.add(symbol)
        i += 1
    return i, code.additions

def main():
    symbols = int(1e4)
    # modes = [round(symbols / 2)]
    # modes = np.arange(1, 400, 10)
    modes = np.arange(1, symbols, 1000)
    results = [lt_simulate(symbols, mode=mode) for mode in modes]
    coded = [result['coded'] for result in results]
    overhead = np.asarray(coded) / symbols
    additions = [result['additions'] / symbols for result in results]
    c = [result['c'] for result in results]

    _ = plt.figure()
    plt.grid(True, which='both')
    plt.ylabel('Overhead', fontsize=18)
    plt.xlabel('$c$', fontsize=18)

    plt.semilogx(c, overhead)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.autoscale(enable=True)
    plt.tight_layout()

    _ = plt.figure()
    plt.grid(True, which='both')
    plt.ylabel('Operations per Source Symbol', fontsize=18)
    plt.xlabel('$c$', fontsize=18)

    plt.semilogx(c, additions)
    # plt.semilogx(parameter, operations_pred)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.autoscale(enable=True)
    plt.tight_layout()

    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
