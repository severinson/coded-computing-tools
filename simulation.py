############################################################################
# Copyright 2016 Albin Severinson                                          #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

'''This module is used to simulate the performance of coded distributed
computing schemes. It connects the solver and evaluation packages and the
complexity module.

'''

import os
import math
import logging
import datetime
import numpy as np
import pandas as pd
import complexity
import model

from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from solvers import Solver
from model import SystemParameters
from assignments.sparse import SparseAssignment
from evaluation import AssignmentEvaluator

# create a process and thread pool executor for this module. these are used to
# increase computations and I/O throughput, respectively.
process_executor = ProcessPoolExecutor()
thread_executor = ThreadPoolExecutor()

def set_load(dataframe, strategy='best'):
    '''compute the communication load for simulated results.

    args:

    strategy: data shuffling strategy. L1, L2, or best. see the paper for
    details.

    '''

    # the load may have already been computed by other means
    if 'load' in dataframe:
        return dataframe

    # otherwise, compute the load depending on what shuffling strategy is used.
    load_1 = dataframe['unicast_load_1'] + dataframe['multicast_load_1']
    load_2 = dataframe['unicast_load_2'] + dataframe['multicast_load_2']
    load_best = pd.concat([load_1, load_2], axis=1).min(axis=1)
    if 'strategy' == 'L1':
        dataframe['load'] = load_1
    elif strategy == 'L2':
        dataframe['load'] = load_2
    elif strategy == 'best':
        dataframe['load'] = load_best
    return dataframe

def flatten_dataframes(dataframe_iter):
    '''flatten an iterable of dataframes by creating a new dataframe where the i-th
    row is the average of all columns from the i-th dataframe in the list.

    '''
    return pd.DataFrame([
        {column:dataframe[column].mean() for column in dataframe}
        for dataframe in dataframe_iter
    ])

def simulate_parameter_list(parameter_list=None, simulate_fun=None,
                            map_complexity_fun=None, encode_delay_fun=None, reduce_delay_fun=None):
    '''Run simulations for a list of parameters.

    args

    parameter_list: list of SystemParameters for which to run simulations.

    simulation_fun: function to apply to each SystemParameters object. use
    functools.partial to set the arguments to the simulate() function below and
    provide it as this argument.

    '''
    assert parameter_list is not None
    assert callable(simulate_fun), simulate_fun
    assert callable(map_complexity_fun), map_complexity_fun
    assert callable(encode_delay_fun), encode_delay_fun
    assert callable(reduce_delay_fun), reduce_delay_fun
    logging.info('Running simulations for %d parameters.', len(parameter_list))

    # run simulations for all parameters. we use a thread pool as most of the
    # time is spent waiting for I/O when loading cached results from disk.
    dataframe_iter = thread_executor.map(simulate_fun, parameter_list)

    # flatten the iterable of dataframes into a single dataframe
    dataframe = flatten_dataframes(dataframe_iter)

    # set the communication load
    dataframe = set_load(dataframe)

    # scale the map phase delay by its complexity
    map_complexity = np.fromiter(
        (map_complexity_fun(parameters) for parameters in parameter_list),
        dtype=float,
    )
    dataframe['delay'] *= map_complexity

    #  compute the encoding and reduce (decoding) delay
    dataframe['encoding'] = np.fromiter(
        (encode_delay_fun(parameters) for parameters in parameter_list),
        dtype=float,
    )
    dataframe['reduce'] = np.fromiter(
        (reduce_delay_fun(parameters) for parameters in parameter_list),
        dtype=float,
    )

    # finally, compute the overall delay
    dataframe['overall_delay'] = dataframe['delay'] + dataframe['encoding'] + dataframe['reduce']

    return dataframe

def parameter_sample(i, parameters=None, parameter_eval=None):
    assert i >= 0 and i % 1 == 0
    assert parameters is not None
    assert parameter_eval is not None
    result = parameter_eval(parameters)
    if isinstance(result, dict):
        return pd.DataFrame(result)
    result['assignment'] = i * np.ones(len(result))
    return result

def assignment_sample(i, parameters=None, solver=None,
                      assignment_eval=None, assignment_type=None):
    assert i >= 0 and i % 1 == 0
    assert parameters is not None
    assert solver is not None
    assert assignment_eval is not None
    assert assignment_type is not None

    # use the solver to find an assignment
    assignment = solver.solve(
        parameters,
        assignment_type=assignment_type
    )

    # make sure the assignment is valid
    if not assignment.is_valid():
        logging.error('Assignment invalid for parameters: %s.', str(parameters))
        return pd.DataFrame()

    # evaluate the performance of the assignment
    result = assignment_eval.evaluate(parameters, assignment)

    if isinstance(result, dict):
        return pd.DataFrame(result)
    result['assignment'] = i * np.ones(len(result))
    return result

def simulate(parameters, directory='./results/', rerun=False, samples=None,
             solver=None, assignment_eval=None, parameter_eval=None,
             assignment_type=None):
    '''simulate a set of system parameters. results are cached on disk.

    the simulator allows for running two kinds of simulations:
        - assigments: create assignments using a solver and evaluate them.
        - parameters: evaluate the performance of a parameters object
          without creating an assignment. useful for analytic performance measures.

    which method is used depends on which on solver, assignment_eval and
    parameter_eval is provided.

    args:

    parameters: SystemParameters to simulate.

    directory: directory to store results in.

    rerun: rerun simulations even if there are results on disk.

    samples: number of samples to simulate.

    solver: assignment solver, i.e., a method that returns a good assignment
    matrix. must be None if a parameter_eval method is provided.

    assignment_eval: method that evaluates the performance of an assignment
    returned by the solver. should be an AssignmentEvaluator object. must be
    provided if a solver is given.

    parameter_eval: method that takes a SystemParameters object and returns a
    DataFrame with its performance. must be None if a solver or assignment_eval
    is provided.

    assignment_type: there are several options for how the assignment matrix is
    stored. this argument sets that type. defaults to SparseAssignment.

    returns: DataFrame with performance samples for all assignments.

    '''
    logging.debug('Running simulations for %s: %s',
                  directory, parameters.identifier())
    assert isinstance(parameters, SystemParameters)
    assert isinstance(directory, str)
    assert samples > 0
    if solver is None:
        assert assignment_eval is None
        assert parameter_eval is not None
    else:
        assert assignment_eval is not None
    if assignment_eval is None:
        assert solver is None
        assert parameter_eval is not None
    else:
        assert solver is not None
    if parameter_eval is None:
        assert solver is not None
        assert assignment_eval is not None
    else:
        assert solver is None
        assert assignment_eval is None

    # there are several options for how the assignment matrix is stored.
    # default to a sparse assignment type
    if assignment_type is None:
        assignment_type = SparseAssignment

    if not os.path.exists(directory):
        os.makedirs(directory)

    # first, attempt to return a cached result
    filename = os.path.join(directory, parameters.identifier() + '.csv')
    if not rerun:
        try:
            dataframe = pd.read_csv(filename)

            # add the system parameters to the dataframe
            for key, value in parameters.asdict().items():
                dataframe[key] = value

            return dataframe
        except FileNotFoundError:
            pass

    best_assignment = None
    best_avg_load = math.inf
    best_avg_delay = math.inf

    printout_interval = datetime.timedelta(seconds=10)
    prev_printout = datetime.datetime.utcnow()

    # select the simulation type
    if solver is None:
        f = partial(
            parameter_sample,
            parameters=parameters,
            parameter_eval=parameter_eval,
        )
    else:
        f = partial(
            assignment_sample,
            parameters=parameters,
            solver=solver,
            assignment_eval=assignment_eval,
            assignment_type=assignment_type,
        )

    # run simulations in parallel using a process pool
    results = process_executor.map(f, range(samples))

    # concatenate the DataFrames and write the result to disk
    dataframe = pd.concat(results)
    dataframe.to_csv(filename)

    # add the system parameters to the dataframe
    for key, value in parameters.asdict().items():
        dataframe[key] = value

    return dataframe
