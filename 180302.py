import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import pyrateless
import plot
import simulation
import rateless
import ita_plots as ita
import complexity
import overhead
import model

from functools import partial

# pyplot settings
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']

def get_parameters():
    '''LT code comparison'''
    rows_per_server = 2000
    rows_per_partition = 10 # optimal number of partitions in this case
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 10
    parameters = list()
    num_servers = 6
    return model.SystemParameters.fixed_complexity_parameters(
        rows_per_server=rows_per_server,
        rows_per_partition=rows_per_partition,
        min_num_servers=num_servers-1,
        code_rate=code_rate,
        muq=muq,
        num_columns=num_columns,
        num_outputs_factor=num_outputs_factor
    )

def map_pdf_plot():

    # semilogy parameters
    num_inputs = 2000
    min_overhead = 1.0
    max_overhead = 1.5
    samples = 1000
    overhead_levels = np.linspace(min_overhead, max_overhead, samples)
    _ = plt.figure(figsize=(10,9))

    # tfp=1e-1
    target_overhead = 1.3
    target_failure_probability = 1e-1

    # find good LT code parameters
    c, delta, mode = rateless.optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )
    mean_degree = pyrateless.Soliton(
        symbols=num_inputs,
        mode=mode,
        failure_prob=delta,
    ).mean()
    print("LT ({}, {}) mean degree={}".format(
        target_overhead,
        target_failure_probability,
        mean_degree,
    ))

    lt_pdf = rateless.lt_success_pdf(
        overhead_levels,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )
    lt_cdf = np.cumsum(lt_pdf)
    # plt.semilogy(overhead_levels, 1-lt_cdf, label='LT $(0.3, 10^{-1})$')

    # tfp=1e-3
    target_overhead = 1.3
    target_failure_probability = 1e-3

    # find good LT code parameters
    c, delta, mode = rateless.optimize_lt_parameters(
        num_inputs=num_inputs,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )
    mean_degree = pyrateless.Soliton(
        symbols=num_inputs,
        mode=mode,
        failure_prob=delta,
    ).mean()
    print("LT ({}, {}) mean degree={}".format(
        target_overhead,
        target_failure_probability,
        mean_degree,
    ))
    lt_pdf = rateless.lt_success_pdf(
        overhead_levels,
        num_inputs=num_inputs,
        mode=mode,
        delta=delta,
    )
    lt_cdf = np.cumsum(lt_pdf)
    # plt.semilogy(overhead_levels, 1-lt_cdf, label='LT $(0.3, 10^{-3})$')

    # random binary fountain
    fountain_pdf = rateless.random_fountain_success_pdf(
        overhead_levels,
        field_size=2,
        num_inputs=num_inputs,
    )
    fountain_cdf = np.cumsum(fountain_pdf)
    plt.semilogy(overhead_levels, 1-fountain_cdf, 'C2', label='$2^{-\delta}$ (R10)')

    # random 256-ary fountain
    fountain_pdf = rateless.random_fountain_success_pdf(
        overhead_levels,
        field_size=256,
        num_inputs=num_inputs,
    )
    fountain_cdf = np.cumsum(fountain_pdf)
    plt.semilogy(overhead_levels, 1-fountain_cdf, 'C3', label='$256^{-\delta}$ (RQ)')

    # failure-overhead curve according to RQ spec.
    overhead = [1, 1+1/num_inputs, 1+2/num_inputs]
    raptorq_cdf = [1/1e2, 1/1e4, 1/1e6]
    plt.semilogy(overhead, raptorq_cdf, 'C4', label='RQ spec.')

    plt.grid()
    # plt.xlim(1.0, 1.5)
    plt.xlim(1.0, 1.025)
    plt.ylim(1e-15, 1.0)
    plt.xlabel("Overhead", fontsize=22)
    plt.ylabel("Failure Probability", fontsize=22)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=25)
    plt.setp(ax.get_yticklabels(), fontsize=25)
    plt.legend(
        numpoints=1,
        shadow=True,
        labelspacing=0,
        columnspacing=0.05,
        fontsize=22,
        fancybox=False,
        borderaxespad=0.1,
    )
    plt.savefig("./plots/180302/failure_overhead_2.png")
    plt.show()
    return

lt_fun = partial(
    simulation.simulate,
    directory='./results/LT_15s_1/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1+15/4000,
        target_failure_probability=1e-1,
    ),
)
lt_fun_r10 = partial(
    simulation.simulate,
    directory='./results/LT_R10/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1+15/4000,
        target_failure_probability=1e-1,
        pdf_fun=partial(
            rateless.random_fountain_success_pdf,
            field_size=2,
        ),
    ),
)
lt_fun_rq = partial(
    simulation.simulate,
    directory='./results/LT_RQ/',
    samples=1,
    parameter_eval=partial(
        rateless.evaluate,
        target_overhead=1+15/4000,
        target_failure_probability=1e-1,
        pdf_fun=partial(
            rateless.random_fountain_success_pdf,
            field_size=256,
        ),
    ),
)

def load_delay_plot():
    parameters = get_parameters()
    print(parameters)
    target_overhead = 1.3
    target_failure_probability = 1e-3

    # find good LT code parameters
    c, delta, mode = rateless.optimize_lt_parameters(
        num_inputs=parameters.num_source_rows,
        target_overhead=target_overhead,
        target_failure_probability=target_failure_probability,
    )

    lt = rateless.performance_integral(
        parameters=parameters,
        num_inputs=parameters.num_source_rows,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
    )

    target_overhead = 1+20/parameters.num_source_rows
    r10 = rateless.performance_integral(
        parameters=parameters,
        num_inputs=parameters.num_source_rows,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
        pdf_fun=partial(
            rateless.random_fountain_success_pdf,
            field_size=2,
        ),
    )
    rq = rateless.performance_integral(
        parameters=parameters,
        num_inputs=parameters.num_source_rows,
        target_overhead=target_overhead,
        mode=mode,
        delta=delta,
        pdf_fun=partial(
            rateless.random_fountain_success_pdf,
            field_size=256,
        ),
    )

    optimal_load = parameters.unpartitioned_load()
    optimal_delay = parameters.computational_delay()
    print("Optimal delay: {} Optimal load: {}".format(optimal_delay, optimal_load))
    print("LT {} / {}".format(lt['delay']/optimal_delay, lt['load']/optimal_load))
    print("R10 {} / {}".format(r10['delay']/optimal_delay, r10['load']/optimal_load))
    print("RQ {} / {}".format(rq['delay']/optimal_delay, rq['load']/optimal_load))
    return

    lt = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_r10 = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun_r10,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    lt_rq = simulation.simulate_parameter_list(
        parameter_list=parameters,
        simulate_fun=lt_fun_rq,
        map_complexity_fun=complexity.map_complexity_unified,
        encode_delay_fun=False,
        reduce_delay_fun=False,
    )
    print(lt)
    print(lt_r10)
    print(lt_rq)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # map_pdf_plot()
    load_delay_plot()


