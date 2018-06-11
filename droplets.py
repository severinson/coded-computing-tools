'''Code relating to the performance of Liquid MapReduce

- Load simulation
- Mean delay should be such that each server has d droplets. This makes more sense.
- DONE: Mean delay assuming perfect knowledge and infinite available computations.
- DONE: Need mean delay assuming limited number of droplets at each server.
- Mean delay assuming randomly selected computations.

'''

import math
import numpy as np
import pynumeric
import stats

from scipy.stats import expon
from scipy.special import lambertw
from numba import jit

class LMR(object):
    '''Liquid MapReduce parameter struct.

    '''
    WORD_SIZE = 8
    ADDITIONC = WORD_SIZE/64
    MULTIPLICATIONC = WORD_SIZE*math.log2(WORD_SIZE)
    def __init__(self, straggling_factor:int=0, decodingf:callable=None, nservers:int=None,
                 nrows:int=None, ncols:int=None, nvectors:int=None,
                 droplet_size:int=1, ndroplets:int=None):
        '''Parameter struct for the Liquid MapReduce system.

        nrows: number of rows of the input matrix.

        ncols: number of columns of the input matrix.

        nvectors: number of input/output vectors. must be divisible by
        the number of servers.

        straggling: scale parameter of the shifted exponential.

        nservers: number of servers.

        droplet_size: number of coded matrix rows in each droplet.

        ndroplets: total number of droplets.

        '''
        if nvectors is None:
            nvectors = nservers
        assert 0 <= straggling_factor < math.inf
        assert 0 < nservers < math.inf
        assert 0 < nrows < math.inf
        assert 0 < ncols < math.inf
        assert 0 < nvectors < math.inf
        if nvectors % nservers != 0:
            raise ValueError('nvectors must be divisible by nservers')
        if ndroplets % nservers != 0:
            raise ValueError('ndroplets must be divisible by nservers')
        self.nrows = nrows
        self.ndroplets = ndroplets
        self.droplet_size = droplet_size
        if self.code_rate > 1:
            raise ValueError('code rate must be <= 1')
        self.ncols = ncols
        self.nvectors = nvectors
        self.nservers = nservers
        self.straggling_factor = straggling_factor
        self.decodingf = decodingf
        return

    @classmethod
    def init2(cls, nrows:int=None, ncols:int=None, nvectors:int=None,
              nservers:int=None, code_rate=None,
              droplets_per_server:int=1, straggling_factor:int=0,
              decodingf:callable=None):
        ndroplets = droplets_per_server*nservers
        droplet_size = nrows / code_rate / nservers / droplets_per_server
        if droplet_size % 1 != 0:
            raise ValueError('droplet_size must be an integer')
        return cls(
            nrows=nrows,
            ncols=ncols,
            nvectors=nvectors,
            nservers=nservers,
            straggling_factor=straggling_factor,
            decodingf=decodingf,
            ndroplets=ndroplets,
            droplet_size=droplet_size,
        )

    def workload(self):
        '''Return the number of additions/multiplications computed by each
        server

        '''
        result = self.droplets_per_server * self.droplet_size
        result *= self.ncols*self.nvectors/self.nservers
        return result

    def asdict(self):
        return {
            'nrows': self.nrows,
            'ncols': self.ncols,
            'nvectors': self.nvectors,
            'nservers': self.nservers,
            'ndroplets': self.ndroplets,
            'droplet_size': self.droplet_size,
            'straggling_factor': self.straggling_factor,
        }

    def __repr__(self):
        return str(self.asdict())

    @property
    def droplets_per_server(self):
        return self.ndroplets / self.nservers

    @property
    def dropletc(self):
        a = (self.ncols - 1) * self.ADDITIONC
        m = self.ncols * self.MULTIPLICATIONC
        return (a+m)*self.droplet_size

    @property
    def decodingc(self):
        return self.decodingf(self)

    @property
    def code_rate(self):
        ncrows = self.ndroplets * self.droplet_size
        return self.nrows / ncrows

    @property
    def straggling(self):
        return self.straggling_factor*self.dropletc

def delays(t, lmr):
    '''Return an array of length <= nservers with simulated delays <= t.

    '''
    a = np.random.exponential(
        scale=lmr.straggling,
        size=lmr.nservers,
    )
    a.sort()
    if t:
        i = np.searchsorted(a, t)
        a = a[:i]
    return a

def drops_estimate(t, lmr):
    '''Return an approximation of the number of droplets computed at time
    t. The inverse of time_estimate.

    '''
    v = -2*lmr.straggling
    v -= lmr.dropletc
    v += (2*lmr.straggling+lmr.dropletc)*math.exp(-t/lmr.straggling)
    v += 2*t
    v /= 2*lmr.dropletc
    return max(lmr.nservers*v, 0)

def delay_estimate(d_tot, lmr):
    '''Return an approximation of the delay t at which d droplets have
    been computed in total over the K servers. The inverse of
    drops_estimate.

    '''
    t = lmr.straggling + lmr.dropletc/2 + d_tot*lmr.dropletc/lmr.nservers
    earg = (lmr.nservers+2*d_tot)*lmr.dropletc
    earg /= -2*lmr.nservers*lmr.straggling
    earg -= 1
    Warg = math.exp(earg)
    Warg *= 2*lmr.straggling + lmr.dropletc
    Warg /= -2*lmr.straggling
    t += lmr.straggling * lambertw(Warg)
    return np.real(t)

def delay_uncoded(lmr):
    '''Return the delay of the uncoded system.

    '''
    rows_per_server = lmr.nrows/lmr.nservers
    result = rows_per_server*(lmr.ncols-1)*lmr.ADDITIONC
    result += rows_per_server*lmr.ncols*lmr.MULTIPLICATIONC
    result *= lmr.nvectors
    result = stats.ShiftexpOrder(
        parameter=lmr.straggling,
        total=lmr.nservers,
        order=lmr.nservers,
    ).mean()
    return result

def delay_mean_empiric(d, lmr, n=100):
    '''Return the simulated mean delay of the map phase.

    Assumes that the map phase ends whenever a total of d droplets
    have been computed and the slowest server is available. Also
    assumes that which droplet to compute next is selected optimally.

    '''
    result = 0
    max_drops = lmr.droplets_per_server*lmr.nvectors
    print(d, max_drops, lmr.nservers, max_drops*lmr.nservers)
    if max_drops * lmr.nservers < d:
        return math.inf
    dropletc = lmr.dropletc # cache this value
    for _ in range(n):
        a = delays(None, lmr)
        t_servers = a[-1] # a is sorted
        f = lambda x: np.floor(
            np.minimum(np.maximum((x-a)/dropletc, 0), max_drops)
        ).sum()
        t_droplets = pynumeric.cnuminv(f, d, tol=lmr.dropletc)
        result += max(t_servers, t_droplets)
    return result/n

def delay_mean_centralized(d, lmr):
    '''Return the mean delay when there is a central reducer, i.e., a
    single master node that does all decoding.

    d: total number of droplets needed. d/nvectors droplets per
    vector.

    '''
    t = np.zeros(lmr.nvectors) # time when decoding of each vector finishes
    d_per_vector = np.ceil(d/lmr.nvectors)
    for i in range(lmr.nvectors):
        t[i] = delay_mean((i+1)*d_per_vector, lmr)
    result = 0
    decodingc = lmr.decodingc # cache
    min_t1 = decodingc*lmr.nvectors
    min_t2 = t[-1]+decodingc
    for i in range(0, lmr.nvectors):
        result = max(result, t[i]) + decodingc

    return result

@jit
def arg_from_order(droplet_order, nvectors, d):
    '''Return the index of droplet_order at which the map phase ends.

    Args:

    droplet_order: array indicating which order the droplets are
    computed in.

    d: required number of droplets per vector.

    '''
    i = 0
    droplets_by_vector = np.zeros(nvectors)
    for v in droplet_order:
        droplets_by_vector[v] += 1
        if droplets_by_vector.min() >= d:
            break
        i += 1
    return i

def delay_mean_empiric_random(d, lmr, n=10):
    '''Return the simulated mean delay of the map phase.

    Assumes that the map phase ends whenever a total of d droplets
    have been computed and the slowest server is available. Which
    droplet to compute is chosen randomly.

    '''
    result = 0
    max_drops = int(np.floor(lmr.droplets_per_server*lmr.nvectors))
    droplet_order = np.zeros(lmr.nservers*max_drops, dtype=int)
    t = np.zeros(lmr.nservers*max_drops)
    if max_drops * lmr.nservers < d:
        return math.inf
    dropletc = lmr.dropletc # cache this value
    server_droplet_order = np.repeat(
        np.arange(lmr.nvectors),
        lmr.droplets_per_server,
    )
    for _ in range(n):
        a = delays(None, lmr)
        assert len(a) == lmr.nservers
        for i in range(lmr.nservers):
            np.random.shuffle(server_droplet_order)
            j1 = int(i*lmr.nvectors*lmr.droplets_per_server)
            j2 = int((i+1)*lmr.nvectors*lmr.droplets_per_server)
            droplet_order[j1:j2] = server_droplet_order[:]
            t[j1:j2] = a[i] + dropletc*np.arange(
                1,
                lmr.nvectors*lmr.droplets_per_server+1,
            )
        p = np.argsort(t)
        droplet_order = droplet_order[p]
        t = t[p]
        i = arg_from_order(droplet_order, lmr.nvectors, d/lmr.nvectors)
        if i >= len(t):
            return math.inf
        t_droplets = t[i]
        t_servers = a[-1] # a is sorted
        print('t_droplets={} t_servers={}'.format(t_droplets, t_servers))
        result += max(t_servers, t_droplets)

    return result/n

def delay_mean(d_tot, lmr):
    '''Return the mean delay of the map phase when d droplets are required
    in total. The returned value is an upper bound on the true mean.

    '''
    t = delay_estimate(d_tot, lmr)
    result = t + lmr.decodingc
    pdf = server_pdf(t, lmr)
    # print()
    # print('t', t)
    for i in range(lmr.nservers-1):
        rv = stats.ShiftexpOrder(
            parameter=lmr.straggling,
            total=lmr.nservers-i-1,
            order=lmr.nservers-i-1,
        )
        # print(pdf[i]*(rv.mean() - lmr.straggling))
        result += pdf[i] * (rv.mean() - lmr.straggling)
    return result

def server_pdf_empiric(t, lmr, n=100000):
    '''Return the PDF over the number of servers with a delay less than t.
    Computed via simulations.

    '''
    pdf = np.zeros(lmr.nservers)
    for _ in range(n):
        i = len(delays(t, lmr))
        pdf[i-1] += 1
    pdf /= n
    return pdf

def server_pdf(t, lmr):
    '''Return the PDF over the number of servers with a delay less than t.

    '''
    pdf = np.zeros(lmr.nservers)
    for i in range(lmr.nservers):
        rv1 = stats.ShiftexpOrder(
            parameter=lmr.straggling,
            order=i+1,
            total=lmr.nservers,
        )
        if i+1 < lmr.nservers:
            rv2 = stats.ShiftexpOrder(
                parameter=lmr.straggling,
                order=i+2,
                total=lmr.nservers,
            )
            pdf[i] = rv1.cdf(t+lmr.straggling)
            pdf[i] -= rv2.cdf(t+lmr.straggling)
        else:
            pdf[i] = rv1.cdf(t+lmr.straggling)
    return pdf

import unittest
class Tests(unittest.TestCase):

    def test_estimates(self):
        '''test that the drops/time estimates are each others inverse'''
        lmr = LMR(
            straggling=100,
            dropletc=10,
            nservers=100,
        )
        t1 = 100
        d = drops_estimate(t1, lmr)
        self.assertGreater(d, 0)
        t2 = delay_estimate(d, lmr)
        self.assertAlmostEqual(t1, t2)
        return

    def test_server_pdf(self):
        '''test the analytic server pdf against simulations'''
        lmr = LMR(
            straggling=100,
            dropletc=10,
            nservers=100,
        )
        t = 100
        pdf_sim = server_pdf_empiric(t, lmr)
        pdf_ana = server_pdf(t, lmr)
        self.assertTrue(np.allclose(pdf_sim, pdf_ana, atol=0.01))
        return

if __name__ == '__main__':
    unittest.main()
