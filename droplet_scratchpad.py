import math
import numpy as np
import matplotlib.pyplot as plt
import stats
import scipy.integrate as integrate
import pynumeric

from scipy.stats import expon
from scipy.special import lambertw
from numba import jit

# pyplot setup
plt.style.use('seaborn-paper')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (4.2, 4.2)
plt.rcParams['figure.dpi'] = 300

'''
- Compare with uncoded, MDS, LT, Raptor

'''

num_servers = 10
complexity = 10
straggling_parameter = 100
# t = 10

# plt.hist(a)
# plt.show()

@jit
def delay_cdf_sim(x, d, n=100):
    '''Evaluate the CDF of the delay at times x.

    Args:

    x: array of delays to evaluate the CDF at.

    d: required number of droplets.

    n: number of samples.

    '''
    cdf = np.zeros(len(x))
    for i, t in enumerate(x):
        for _ in range(n):
            a = get_delay(math.inf)
            drops = np.floor((t-a)/complexity).sum()
            if drops >= d and a[-1] <= t:
                cdf[i] += 1
        cdf[i] /= n
    return cdf

def droplet_mean_delay(d):
    '''Return the mean delay of the map phase.

    '''
    t = time_estimate(d)
    result = t
    pdf = server_pdf(t)
    for i in range(num_servers-1):
        rv = stats.ShiftexpOrder(
            parameter=straggling_parameter,
            total=num_servers-i-1,
            order=num_servers-i-1,
        )
        result += pdf[i] * (rv.mean() - straggling_parameter)
    return result

def droplet_mean_load_sim():
    '''Return the mean communication load. Based on simulations.

    '''

def plot_cdf(num_droplets=2000):
    # x = np.linspace(0, 100*max(straggling_parameter, complexity), 10)
    # time needed to get the droplets
    t = pynumeric.cnuminv(drops_estimate, target=num_droplets)
    plt.plot([t, t], [0, 1], label='avg. t')

    # simulated
    x = np.linspace(t/1.05, t*1.05, 100)
    cdf = delay_cdf_sim(x, num_droplets)
    plt.plot(x, cdf, label='simulation')

    # gamma cdf
    rv = stats.ShiftexpOrder(
        parameter=straggling_parameter,
        total=num_servers,
        order=num_servers,
    )
    t_cdf = rv.mean()-straggling_parameter
    plt.plot([t_cdf, t_cdf], [0, 1], label='cdf t')

    pdf = np.diff(cdf)
    pdf /= pdf.sum()
    mean_t = integrate.trapz(pdf*x[1:])
    # mean_t = (pdf*x[1:]).sum()

    t_ana = delay_mean(num_droplets)
    plt.plot([t_ana, t_ana], [0, 1], label='analytic')

    print('empric: {} finv: {} cdf: {} analytic: {}'.format(mean_t, t, t_cdf, t_ana))

    # only order statistics
    # cdf = delay_cdf(x, 1000)
    # plt.plot(x, cdf, label='order statistic')
    plt.grid()
    plt.legend()
    # plt.show()
    return

def drops_simulation(t, n=100):
    result = 0
    for _ in range(n):
        a = get_delay(t)
        tmp = np.floor((t-a)/complexity)
        if len(tmp) == 0:
            continue
        assert tmp.min() >= 0
        result += tmp.sum()

    return result/n

def server_pdf_empiric(t, n=100000):
    '''Return the PDF over the number of servers with a delay less than t.
    Computed via simulations.

    '''
    pdf = np.zeros(num_servers)
    for _ in range(n):
        i = len(get_delay(t))
        pdf[i-1] += 1
    pdf /= n
    return pdf

def server_pdf(t):
    '''Return the PDF over the number of servers with a delay less than t.

    '''
    pdf = np.zeros(num_servers)
    for i in range(num_servers):
        rv1 = stats.ShiftexpOrder(
            parameter=straggling_parameter,
            order=i+1,
            total=num_servers,
        )
        if i+1 < num_servers:
            rv2 = stats.ShiftexpOrder(
                parameter=straggling_parameter,
                order=i+2,
                total=num_servers,
            )
            pdf[i] = rv1.cdf(t+straggling_parameter)
            pdf[i] -= rv2.cdf(t+straggling_parameter)
        else:
            pdf[i] = rv1.cdf(t+straggling_parameter)
    return pdf

def time_estimate_K1(d):
    '''Return an approximation of the time t at which d droplets have been
    computed. The inverse of drops_estimate.

    '''
    t = straggling_parameter + complexity/2 + d*complexity
    earg = 2*straggling_parameter
    earg += complexity + 2*d*complexity
    earg /= -2*straggling_parameter
    Warg = math.exp(earg)
    Warg *= -2*straggling_parameter - complexity
    Warg /= 2*straggling_parameter
    t += straggling_parameter * lambertw(Warg)
    return t

def test_estimates(t1=1000):
    d = drops_estimate(t1)
    t2 = time_estimate(d)
    print(t1, t2, d)
    return

def bound4(t):
    bf = lambda x: math.floor((t-x) / complexity)
    f = lambda x: expon.pdf(x, scale=straggling_parameter)*bf(x)
    v, _ = integrate.quad(f, 0, t)
    ub = num_servers * v
    lb = max(ub-num_servers+1, 0)
    return lb, ub

def bound3(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    for i in range(num_servers):
        norm = expon.cdf(t, scale=straggling_parameter)
        v = 1/straggling_parameter*t
        v += math.exp(-1/straggling_parameter*t) - 1
        v *= straggling_parameter
        v /= norm * complexity
        ub = v*(i+1)
        # lb = max(ub-i+1, 0)
        lb = ub-i+1
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb
    return lbt, ubt

def bound2(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    bf = lambda x: (t-x) / complexity
    for i in range(num_servers):
        norm = expon.cdf(t, scale=straggling_parameter)
        f = lambda x: expon.pdf(x, scale=straggling_parameter)/norm*bf(x)
        if norm < np.finfo(float).tiny:
            continue
        v, _ = integrate.quad(f, 0, t)
        # ub = math.floor(v)*(i+1)
        ub = v*(i+1)
        lb = max(ub-i+1, 0)
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb
    return lbt, ubt

def bound(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    # print()
    for i in range(num_servers):
        rv = stats.ExpSum(
            scale=straggling_parameter,
            order=i+1,
        )
        norm = rv.cdf((i+1)*t)
        # s = integrate.quad(lambda x: rv.pdf(x)/norm, 0, (i+1)*t)
        # print('sum={}, norm={}'.format(s, norm))
        if norm < np.finfo(float).tiny:
            continue

        # bf = lambda x: ((i+1)*t-x)/complexity
        # p, _ = integrate.quad(rv.pdf, 0, straggling_parameter)
        # print(p, )
        # ub, _ = integrate.quad(
        #     lambda x: rv.pdf(x)/norm * bf(x),
        #     0,
        #     (i+1)*t,
        # )
        ub = 0
        if t > straggling_parameter:
            ub += (i+1)*(t - straggling_parameter)/complexity

        lb = max(ub-i+1, 0)
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb

    return lbt, ubt

def drops_cdf():
    t = np.linspace(1, 10*max(straggling_parameter, complexity), 100)
    y = [computed_drops(i) for i in t]
    plt.plot(t, [i[0] for i in y], '-o', label='simulation', markevery=0.1)
    # plt.plot(t, [i[1] for i in y], label='lb')
    # plt.plot(t, [i[2] for i in y], label='up')
    # plt.plot(t, [(i[1]+i[2])/2 for i in y], label='(lb+up)/2')

    y = [bound4(i) for i in t]
    # plt.plot(t, [i[0] for i in y], label='analytic lb')
    plt.plot(t, [i[1] for i in y], '-', label='integral')
    # plt.plot(t, [(i[0]+i[1])/2 for i in y], label='analytic mean')

    y = [bound5(i) for i in t]
    plt.plot(t, [i[1] for i in y], '--', label='approximation')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('avg. number of droplets computed')
    plt.legend()
    plt.show()
    return

def estimate_waste():
    a = get_delay(verbose=False)
    d = np.floor((t-a)/complexity)
    assert d.min() >= 0
    waste = t - a - d*complexity
    est = complexity/2
    print('mean waste is {}. est is {}'.format(waste.mean(), est))
    print('{} droplets computed'.format(d.sum()))
    plt.hist(waste,bins=50)
    plt.show()
    return

def estimate_hist():
    # d, est1, est2 = estimates()
    samples = np.fromiter((estimates()[0] for _ in range(1000)), dtype=int)
    est1 = np.fromiter((estimates()[1] for _ in range(1000)), dtype=int)
    est2 = np.fromiter((estimates()[2] for _ in range(1000)), dtype=int)
    print('true mean is', samples.mean())
    # print('est1 mean is', est1.mean())
    print('est2 mean is', est2.mean())

    plt.figure()
    plt.hist(samples, label='true')
    # plt.hist(est1, label='est1')
    plt.hist(est2, label='est2')
    plt.legend()
    plt.show()
    return

def delay_samples(t):
    samples = list()
    lsamples = list()
    for _ in range(10000):
        a = get_delay(t)
        samples.append(a.sum())
        lsamples.append(len(a))
    return np.fromiter(samples, dtype=float), np.fromiter(lsamples, dtype=float)

def delay_hist():
    # it's probably right that the exp rvs just scale. but the gamma
    # does something else. we should maybe just settle for the
    # asymptotic expression. it does pretty well..
    t = 0.5*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=0.5')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 1*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=1')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 2*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=2')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 3*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=3')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 4*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=4')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    plt.legend()
    plt.grid()
    plt.show()
    return

def plot_integrand():
    t = 100*straggling_parameter
    bf = lambda x: math.floor((t-x) / complexity)
    bf2 = lambda x: (t-x) / complexity
    f = lambda x: expon.pdf(x, scale=straggling_parameter)*bf(x)
    x = np.linspace(0, t, 10)
    y = [bf(i) for i in x]
    # plt.plot(x, y, label='floor')
    y2 = [bf2(i) for i in x]
    # plt.plot(x, y2, label='no floor')
    plt.plot(x, [i2-i1 for i1, i2 in zip(y, y2)], label='diff')

    dx1 = straggling_parameter+complexity
    dy1 = bf2(dx1) - bf(dx1)
    a1 = dy1/dx1
    f = lambda x: a1*x
    x = np.linspace(0, straggling_parameter+complexity)
    plt.plot(x, [f(i) for i in x], '--')

    dx2 = t
    dy2 = 0
    a2 = -dy1 / (dx2-dx1)
    c = dy1 - a2*dx1
    f = lambda x: a2*x+c
    x = np.linspace(straggling_parameter+complexity, t)
    plt.plot(x, [f(i) for i in x], '--')

    plt.grid()
    plt.legend()
    plt.show()
    return

def exptest():
    '''Plots to investigate if an exponential RV truncated on the left is
    a shifted exponential.

    '''
    trunc = 2
    norm = 1-expon.cdf(trunc, scale=straggling_parameter)
    print(norm)
    t = np.linspace(0, trunc)
    tt = np.linspace(trunc, 2*trunc)
    plt.plot(t, expon.pdf(tt, scale=straggling_parameter)/norm, label='shifted')
    plt.plot(t, expon.pdf(t, scale=straggling_parameter), label='orig')
    plt.grid()
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    # delay_hist()
    # order = 100
    # rv = stats.ExpSum(
    #     scale=straggling_parameter,
    #     order=num_servers,
    # )
    # t = 1*straggling_parameter
    # norm = rv.cdf(t)
    # samples = list()
    # for _ in range(10000):
    #     # x = np.random.exponential(scale=straggling_parameter, size=order).sum()
    #     # samples.append(x)
    #     samples.append(get_delay(t).sum())
    # samples = np.fromiter(samples, dtype=float)
    # plt.hist(samples, bins=50, density=True)
    # t = np.linspace(samples.min(), samples.max())
    # plt.plot(t, [rv.pdf(i) for i in t])
    # plt.grid()
    # plt.show()

    # print('analytic={} simulated={}'.format(rv.mean(), m))
    # pdf = server_pdf(t)
    # drops_cdf()
    # estimate_waste()
    # estimate_hist()
    # plot_integrand()
    plot_cdf()
    # exptest()
    # test_estimates()
