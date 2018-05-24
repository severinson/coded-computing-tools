'''Plotting tools

'''

import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.special import comb as nchoosek

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
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns
        )
        parameters.append(par)
    return parameters

def get_parameters_size_2():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 10
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 1000
    # num_outputs_factor = 10
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def get_parameters_size_3():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 100
    code_rate = 2/3
    muq = 2
    num_columns = None
    # num_outputs_factor = 1000
    num_outputs_factor = 10
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            rows_per_partition=rows_per_partition,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def get_parameters_size_4():
    '''Get a list of parameters for the size plot.'''
    rows_per_server = 2000
    rows_per_partition = 100
    code_rate = 2/3
    muq = 2
    num_columns = None
    num_outputs_factor = 1000
    parameters = list()
    num_servers = [5, 8, 20, 50, 80, 125, 200, 500, 2000]
    for servers in num_servers:
        par = model.SystemParameters.fixed_complexity_parameters(
            rows_per_server=rows_per_server,
            min_num_servers=servers,
            code_rate=code_rate,
            muq=muq,
            num_columns=num_columns,
            num_outputs_factor=num_outputs_factor
        )
        parameters.append(par)
    return parameters

def get_parameters_tradeoff():
    '''Get a list of parameters for the load-vs-delay plot.'''
    num_outputs = 840
    num_servers = 21
    server_storage = 1/2
    parameters = list()
    num_source_rows = 352716
    for q in range(8, num_servers+1):
        num_coded_rows = num_source_rows * num_servers / q
        num_batches = nchoosek(num_servers, int(server_storage*q))
        rows_per_batch = int(num_coded_rows / num_batches)
        try:
            par = model.SystemParameters(
                rows_per_batch=1,
                num_servers=num_servers,
                q=q,
                num_outputs=q,
                server_storage=server_storage,
                num_partitions=1,
            )
        except ValueError:
            continue
        parameters.append(par)
    return parameters

def get_parameters_N():
    '''Get a list of parameters for the N to n ratio plot.'''
    rows_per_batch = 100
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/3
    num_partitions = 240
    num_outputs = 10*q
    # num_columns = 20000
    parameters = list()
    for i in range(1, 11):
        # num_outputs = i * q
        num_columns = pow(i, 2) * 200
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=num_partitions,
            num_columns=num_columns,
        )
        parameters.append(par)

    return parameters

def get_parameters_deadline():
    '''Get a list of parameters for the N to n ratio plot.'''
    rows_per_batch = 10
    num_servers = 201
    q = 134
    # num_partitions = [
    #     100, 200, 268,
    # ]
    # num_partitions = rows_per_batch
    num_partitions = 8375

    # num_columns = 5000
    # num_columns = 10000
    num_columns = None
    # for T in num_partitions:
    num_outputs = 1340
    server_storage = 2/q
    parameters = model.SystemParameters(
        rows_per_batch=rows_per_batch,
        num_servers=num_servers,
        q=q,
        num_outputs=num_outputs,
        server_storage=server_storage,
        num_partitions=num_partitions,
        num_columns=num_columns,
    )
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
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=partitions,
        )
        parameters.append(par)

    return parameters

def get_parameters_partitioning_2():
    '''Constant system size, increasing partitions, num_outputs=num_columns'''
    rows_per_batch = 250
    num_servers = 9
    q = 6
    num_outputs = 6000
    server_storage = 1/3
    num_partitions = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30,
                      40, 50, 60, 75, 100, 120, 125, 150, 200, 250,
                      300, 375, 500, 600, 750, 1000, 1500, 3000]

    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=partitions,
        )
        parameters.append(par)

    return parameters

def get_parameters_partitioning_3():
    '''Get a list of parameters for the size plot.'''
    rows_per_batch = 10
    num_servers = 201
    q = 134
    num_outputs = 1340
    server_storage = 2 / q
    # num_partitions = [2]
    num_partitions = [10, 134, 8375, 13400, 16750, 67000]
    # 26800
    parameters = list()
    for partitions in num_partitions:
        par = model.SystemParameters(
            rows_per_batch=rows_per_batch,
            num_servers=num_servers,
            q=q,
            num_outputs=num_outputs,
            server_storage=server_storage,
            num_partitions=partitions,
        )
        parameters.append(par)

    return parameters


def load_delay_plot(results, plot_settings, xdata, xlabel='',
                    normalize=None, legend='load', ncol=1, loc='best',
                    ylim_top=None, xlim_top=None,
                    ylim_bot=None, xlim_bot=None,
                    vline=None, show=True):
    '''Create a plot with two subplots for load and delay respectively.

    Args:

    results: List of Dataframes.

    plot_settings: List of dicts with plot settings.

    xdata: Label of the X axis data ('partitions' or 'servers').

    xlabel: X axis label

    normalize: If a SimulatorResult is provided, all ploted results
    are normalized by this one.

    legend: Place the legend in the load or delay plot by setting this
    argument to 'load' or 'delay'.

    ncol: number of legend columns.

    loc: legend location.

    ylim_top, xlim_bot, ylim_bot, xlim_bot: tuples (min, max) used to set axis
    limits for y/x for the top and bottom plots. Set to None to use default
    axis limits.

    vline: plot a vertical line at this value.

    show: show the plots if True.

    '''
    assert isinstance(results, list)
    assert isinstance(plot_settings, list)
    # assert isinstance(normalize, SimulatorResult) or normalize is None
    assert isinstance(show, bool)

    # plt.rc('pgf',  texsystem='pdflatex')
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    # _ = plt.figure(figsize=(10,9))
    # _ = plt.figure(figsize=(11,4))
    plt.figure()

    # Plot load
    plt.autoscale(enable=True)
    plt.tight_layout()
    ax1 = plt.subplot(211)
    plt.setp(ax1.get_xticklabels(), visible=False)
    for result, plot_setting in zip(results, plot_settings):
        markevery = 0.2
        if 'markevery' in plot_setting:
            markevery = plot_setting['markevery']

        plot_result(
            result,
            plot_setting,
            xdata,
            'load',
            ylabel=r'$L$',
            subplot=True,
            normalize=normalize,
            markevery=markevery,
        )

    # plot a vertical bar at the partitioning limit
    if vline:
        plt.axvline(x=vline, color='k', linestyle=':', linewidth=3)

    plt.margins(y=0.1)
    if legend == 'load':
        plt.legend(
            numpoints=1,
            shadow=True,
            labelspacing=0,
            loc=loc,
            fancybox=False,
            borderaxespad=0.1,
            ncol=ncol,
        )

    # Plot delay
    ax2 = plt.subplot(212, sharex=ax1)
    for result, plot_setting in zip(results, plot_settings):
        markevery = 0.2
        if 'markevery' in plot_setting:
            markevery = plot_setting['markevery']

        plot_result(
            result,
            plot_setting,
            xdata,
            'overall_delay',
            xlabel=xlabel,
            ylabel=r'$D$',
            subplot=True,
            normalize=normalize,
            markevery=markevery,
        )

    if legend == 'delay':
        plt.legend(
            numpoints=1,
            shadow=True,
            labelspacing=0,
            loc=loc,
            fancybox=False,
            borderaxespad=0.1,
            ncol=ncol,
        )

    # plot a vertical bar at the partitioning limit
    if vline:
        plt.axvline(x=vline, color='k', linestyle=':', linewidth=3)

    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.12)
    plt.margins(y=0.1)

    # set axis limits
    if ylim_top:
        ax1.set_ylim(ylim_top)
    if xlim_top:
        ax1.set_xlim(xlim_top)
    if ylim_bot:
        ax2.set_ylim(ylim_bot)
    if xlim_bot:
        ax2.set_xlim(xlim_bot)

    if show:
        plt.show()
    return

def encode_decode_plot(results, plot_settings, xdata, xlabel='',
                       normalize=None, legend='encode', ncol=1, loc='best',
                       ylim_top=None, xlim_top=None,
                       ylim_mid=None, xlim_mid=None,
                       ylim_bot=None, xlim_bot=None,
                       show=True):
    '''Create a plot with two subplots for encoding and decoding delay respectively.

    args:

    results: list of Dataframes.

    plot_settings: List of dicts with plot settings.

    xdata: Label of the X axis data ('partitions' or 'servers').

    xlabel: X axis label

    normalize: If a SimulatorResult is provided, all ploted results
    are normalized by this one.

    legend: Place the legend in the load or delay plot by setting this
    argument to 'load' or 'delay'.

    ylim_top, xlim_bot, ylim_bot, xlim_bot: tuples (min, max) used to set axis
    limits for y/x for the top and bottom plots. Set to None to use default
    axis limits.

    show: show the plots if True.

    '''
    assert isinstance(results, list)
    assert isinstance(plot_settings, list)
    assert isinstance(show, bool)

    # plt.rc('pgf',  texsystem='pdflatex')
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
    # _ = plt.figure(figsize=(10,9))
    plt.figure()

    # encode delay
    plt.autoscale(enable=True)
    plt.tight_layout()
    ax1 = plt.subplot(311)
    plt.setp(ax1.get_xticklabels(), visible=False)
    for df, plot_setting in zip(results, plot_settings):
        df = df.copy()
        if normalize is not None:
            df['encode'] /= normalize['overall_delay']
        else:
            df['encode'] /= df['overall_delay']
        plot_result(
            df,
            plot_setting,
            xdata,
            'encode',
            ylabel='Encoding Delay',
            subplot=True,
        )

    plt.margins(y=0.1)
    plt.title("encode / reduce / map")
    if legend == 'encode':
        plt.legend(
            numpoints=1,
            shadow=True,
            labelspacing=0,
            loc=loc,
            fancybox=False,
            borderaxespad=0.1,
            ncol=ncol,
        )

    # reduce/decode delay
    ax2 = plt.subplot(312, sharex=ax1)
    for df, plot_setting in zip(results, plot_settings):
        df = df.copy()
        if normalize is not None:
            df['reduce'] /= normalize['overall_delay']
        else:
            df['reduce'] /= df['overall_delay']
        plot_result(
            df,
            plot_setting,
            xdata,
            'reduce',
            xlabel=xlabel,
            ylabel='Decoding Delay',
            subplot=True,
        )

    if legend == 'decode':
        plt.legend(
            numpoints=1,
            shadow=True,
            labelspacing=0,
            loc=loc,
            fancybox=False,
            borderaxespad=0.1,
            ncol=ncol,
        )

    # map delay
    ax3 = plt.subplot(313, sharex=ax1)
    for df, plot_setting in zip(results, plot_settings):
        df = df.copy()
        if normalize is not None:
            df['delay'] /= normalize['overall_delay']
        else:
            df['delay'] /= df['overall_delay']
        plot_result(
            df,
            plot_setting,
            xdata,
            'delay',
            xlabel=xlabel,
            ylabel='Map Delay',
            subplot=True,
        )

    if legend == 'delay':
        plt.legend(
            numpoints=1,
            shadow=True,
            labelspacing=0,
            loc=loc,
            fancybox=False,
            borderaxespad=0.1,
            ncol=ncol,
        )

    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.margins(y=0.1)

    # set axis limits
    if ylim_top:
        ax1.set_ylim(ylim_top)
    if xlim_top:
        ax1.set_xlim(xlim_top)
    if ylim_mid:
        ax2.set_ylim(ylim_mid)
    if xlim_mid:
        ax2.set_xlim(xlim_mid)
    if ylim_bot:
        ax3.set_ylim(ylim_bot)
    if xlim_bot:
        ax3.set_xlim(xlim_bot)

    if show:
        plt.show()
    return

def plot_result(result, plot_settings, xdata, ydata, xlabel='',
                ylabel='', subplot=False, normalize=None,
                errorbars=False, plot_type='semilogx',
                markevery=None):
    '''Plot simulated results.

    Args:

    result: A SimulatorResult.

    plot_settings: A dict with plot settings.

    xdata: Label of the X axis data ('partitions' or 'servers').

    ydata: Label of the Y axis data ('load' or 'delay').

    xlabel: X axis label.

    ylabel: Y axis label.

    subplot: Set to True if the plot q is intended to be a subplot.
    This will keep it from creating a new plot window, creating a
    legend, and automatically showing the plot.

    normalize: Normalize the plotted data by that of these results.
    Must be a list of SimulationResults of length equal to results.

    errorbars: Plot error bars.

    '''
    assert isinstance(plot_settings, dict)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(subplot, bool)

    if not subplot:
        _ = plt.figure()

    plt.grid(True, which='both')
    plt.autoscale()

    label = plot_settings['label']
    color = plot_settings['color']
    style = color + plot_settings['marker']

    xarray = result[xdata]
    ymean = result[ydata].copy()
    ymin = ymean.copy()
    ymax = ymean.copy()
    yerr = np.zeros([2, len(ymean)])
    yerr[0, :] = ymean - ymin
    yerr[1, :] = ymax - ymean
    if normalize is not None:
        ymean /= normalize[ydata]

    if plot_type == 'semilogx':
        plt.semilogx(xarray, ymean, style, label=label, markevery=markevery)
    elif plot_type == 'loglog':
        plt.loglog(xarray, ymean, style, label=label, markevery=markevery)

    if errorbars:
        plt.errorbar(xarray, ymean, yerr=yerr, fmt='none', ecolor=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not subplot:
        plt.legend(numpoints=1, loc='best', prop={'weight': 'bold'})
        plt.show()

    return
