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

'''A simulator is created by providing it with an assignment type, a
solver, and an evaluation function. Once created it allows for running
simulations for sets of parameters. The module automatically handles
saving/loading results and assignments to disk.

'''

import os
import math
import logging
import datetime
import numpy as np
import pandas as pd
import complexity
import model

from concurrent.futures import ProcessPoolExecutor
from solvers import Solver
from model import SystemParameters
from assignments.sparse import SparseAssignment
from evaluation import AssignmentEvaluator

class SimulatorError(Exception):
    '''Base class for exceptions thrown by this module.'''

class SimulatorResult(object):
    '''This object represents the result of a simulation. It transparently
    handles saving/loading results to/from disk.

    '''

    def __init__(self, simulator, parameter_list, directory):
        '''Create a simulator result.

        Args:

        simulator: The simulator creating the result.

        parameter_list: List of system paramaters the result applies
        to.

        directory: Results directory name.

        '''
        assert isinstance(simulator, Simulator)
        assert isinstance(parameter_list, list)
        assert isinstance(directory, str)
        self.simulator = simulator
        self.parameter_list = parameter_list
        self.directory = directory

        # Try to load results from disk
        self.dataframes = list()
        for parameters in parameter_list:
            filename = os.path.join(directory, parameters.identifier() + '.csv')
            try:
                self.dataframes.append(pd.read_csv(filename))
            except FileNotFoundError:
                self.dataframes.append(None)

        # Set default parameters
        self.set_uncoded(enable=False)
        self.set_cmapred(enable=False)
        self.set_stragglerc(enable=False)
        self.set_encode_delay()
        self.set_reduce_delay()
        self.set_shuffling_strategy()
        return

    def __getitem__(self, key):
        '''Get data arrays from the simulated results.'''
        if key == 'load':
            return self.load()

        if key == 'delay':
            return self.delay()

        if key == 'encode':
            return self.encode_delay()

        if key == 'reduce':
            return self.reduce_delay()

        if key == 'partitions':
            return np.asarray([parameters.num_partitions
                               for parameters in self.parameter_list])

        if key == 'servers':
            return np.asarray([parameters.num_servers
                               for parameters in self.parameter_list])

        if key == 'num_inputs':
            return np.asarray(
                [parameters.num_outputs for parameters in self.parameter_list]
            )

        if key == 'num_columns':
            return np.asarray(
                [parameters.num_columns for parameters in self.parameter_list]
            )
        raise SimulatorError('No data for key {}.'.format(key))

    def append_results(self, index, dataframe):
        '''Append performance samples.

        Args:

        index: Prameter index of the result.

        dataframe: Pandas DataFrame with results to append. Must
        contain the same columns as any already stored results.

        '''
        assert isinstance(index, int) and 0 <= index < len(self.dataframes)
        if self.dataframes[index] is None:
            self.dataframes[index] = dataframe
        else:
            pass
        # self.dataframes[index].append(dataframe)

        # Write results to disk
        filename = os.path.join(self.directory,
                                self.parameter_list[index].identifier() + '.csv')
        self.dataframes[index].to_csv(filename)
        return

    def set_uncoded(self, enable=True):
        '''Compute load and delay for these parameters as if it's uncoded.'''
        self.uncoded = enable
        return

    def set_cmapred(self, enable=True):
        '''Compute load and delay for these parameters using no erasure code
        to deal with stragglers, i.e., only coded MapReduce.

        '''
        self.cmapred = enable
        return

    def set_stragglerc(self, enable=True):
        '''Compute load/delay for a system using only straggler coding, i.e.,
        using an erasure code to deal with stragglers but no coded
        multicasting.

        '''
        self.stragglerc = enable
        return

    def set_encode_delay(self, function=None):
        '''Include the encoding delay in the total computational delay.

        Args:

        function: A function that takes a parameters object and returns its
        encoding delay.

        '''
        logging.debug('Set encode function %s.', str(function))
        self.encode_function = function
        return

    def set_reduce_delay(self, function=None):
        '''Include the reduce (decoding) delay in the total computational delay.

        Args:

        function: A function that takes a parameters object and
        returns its reduce delay.

        '''
        logging.debug('Set reduce function %s.', str(function))
        self.reduce_function = function
        return

    def set_shuffling_strategy(self, strategy='best'):
        '''Set the data shuffling strategy.

        Args:

        strategy: Can be L1, L2, or 'best.

        '''
        assert strategy == 'L1' or strategy == 'L2' or strategy == 'best'
        self.shuffling_strategy = strategy
        return

    def load(self):
        '''Collect the communication load of all parameters in this result
        into a single array.

        '''
        # loads = np.zeros([3, len(self.dataframes)])
        loads = np.zeros(len(self.dataframes))
        for i in range(len(self.dataframes)):
            parameters = self.parameter_list[i]
            dataframe = self.dataframes[i]
            if dataframe is None:
                loads.append(math.inf)
                continue

            if 'load' in dataframe:
                frame_load = dataframe['load'].copy()
            elif self.shuffling_strategy == 'L1':
                frame_load = dataframe['unicast_load_1'] + dataframe['multicast_load_1']
            elif self.shuffling_strategy == 'L2':
                frame_load = dataframe['unicast_load_2'] + dataframe['multicast_load_2']
            elif (dataframe['unicast_load_1'].mean() + dataframe['multicast_load_1'].mean() <
                  dataframe['unicast_load_2'].mean() + dataframe['multicast_load_2'].mean()):
                frame_load = dataframe['unicast_load_1'] + dataframe['multicast_load_1']
            else:
                frame_load = dataframe['unicast_load_2'] + dataframe['multicast_load_2']

            loads[i] = frame_load.mean()
            # loads[0, i] = frame_load.mean()
            # loads[1, i] = frame_load.min()
            # loads[2, i] = frame_load.max()

        return loads

    def encode_delay(self):
        '''Collect the encoding delay of all parameters into a single array. This
        method is used to collect only the encoding delay. It's not used by the delay
        method as that method does its own scaling.

        '''
        assert self.encode_function is not None, \
            'Encode function must be set before computing encoding delay: %s' % self.directory

        delays = np.fromiter(
            (self.encode_function(parameters) for parameters in self.parameter_list),
            dtype=float,
        )
        delays /= np.fromiter(
            (parameters.num_source_rows for parameters in self.parameter_list),
            dtype=float,
        )
        return delays

    def reduce_delay(self):
        '''Collect the reduce (decoding) delay of all parameters into a single array.
        This method is used to collect only the reduce (decoding) delay. It's
        not used by the delay method as that method does its own scaling.

        '''
        assert self.reduce_function is not None, \
            'Reduce function must be set before computing reduce delay: %s' % self.directory
        delays = np.fromiter(
            (self.reduce_function(parameters) for parameters in self.parameter_list),
            dtype=float,
        )
        delays /= np.fromiter(
            (parameters.num_source_rows for parameters in self.parameter_list),
            dtype=float,
        )
        return delays

    def delay(self):
        '''Collect the computational delay of all parameters in this result
        into a single array.

        '''

        delays = np.zeros(len(self.dataframes))
        for i in range(len(self.dataframes)):
            parameters = self.parameter_list[i]
            dataframe = self.dataframes[i]
            if dataframe is None:
                delays.append(math.inf)
                continue

            frame_delay = dataframe['delay'].copy()

            # Uncoded systems are handled separately
            if self.uncoded:
                uncoded_storage = 1 / parameters.num_servers
                rows_per_server = uncoded_storage * parameters.num_source_rows
                frame_delay *= complexity.matrix_vector_complexity(
                    rows_per_server,
                    parameters.num_columns
                )

            # If no straggler erasure code is used, i.e., only coded MapReduce.
            elif self.cmapred:
                server_storage = parameters.muq / parameters.num_servers
                rows_per_server = server_storage * parameters.num_source_rows
                frame_delay *= complexity.matrix_vector_complexity(
                    rows_per_server,
                    parameters.num_columns
                )

            # If only straggler coding is used, i.e., no coded multicasts.
            elif self.stragglerc:
                server_storage = 1 / parameters.q
                rows_per_server = server_storage * parameters.num_source_rows
                frame_delay *= complexity.matrix_vector_complexity(
                    rows_per_server,
                    parameters.num_columns
                )

            else:
                rows_per_server = parameters.server_storage * parameters.num_source_rows
                frame_delay *= complexity.matrix_vector_complexity(
                    rows_per_server,
                    parameters.num_columns
                )

            # each server multiplies the rows it stores by all vectors
            frame_delay *= parameters.num_outputs

            # include reduce time if enabled. this value must not be normalized.
            if self.reduce_function is not None:
                frame_delay += self.reduce_function(parameters)

            # include encode time if enabled. this value must not be normalized.
            if self.encode_function is not None:
                frame_delay += self.encode_function(parameters)

            # normalize and append
            frame_delay /= parameters.num_source_rows
            delays[i] = frame_delay.mean()
            # delays[0, i] = frame_delay.mean()
            # delays[1, i] = frame_delay.min()
            # delays[2, i] = frame_delay.max()

        return delays

class Simulator(object):
    '''This object connects the assignments, solvers, and evaluation
    packages.

    '''

    def __init__(self, solver=None, assignment_eval=None,
                 parameter_eval=None, assignment_type=None,
                 directory='./results/', assignments=1, rerun=False):
        '''Create a simulator.

        The simulator allows for running two kinds of simulations:
        - assigments: Create assignments using a solver and evaluate it.
        - parameters: Evaluate the performance of a parameters object
          without creating an assignment.

        Args:

        solver: An assignment solver. Must be None if a parameter_eval
        method is provided.

        assignment_eval: AssignmentEvaluator object. Must be provided
        if a solver is.

        parameter_eval: A function that takes a SystemParameters
        object and returns a DataFrame with its performance. Must be
        None if a solver or assignment_eval is provided.

        assignment_type: Type of assignment matrix to return. Defaults
        to SparseAssignment.

        directory: Store results in this folder.

        assignments: Number of assignments to simulate when running
        assignment simulations.

        rerun: Simulations are re-run even if there are results disk
        if this is True.

        '''
        if parameter_eval is None:
            assert isinstance(solver, Solver)
            assert isinstance(assignment_eval, AssignmentEvaluator)
        else:
            assert solver is None
            assert assignment_eval is None

        if assignment_eval is None:
            pass
        else:
            assert isinstance(solver, Solver)
            assert isinstance(assignment_eval, AssignmentEvaluator)
            assert parameter_eval is None

        assert isinstance(directory, str)
        assert isinstance(assignments, int) and assignments > 0
        assert isinstance(rerun, bool)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.solver = solver
        self.assignment_eval = assignment_eval
        self.parameter_eval = parameter_eval

        if assignment_type is None:
            self.assignment_type = SparseAssignment
        else:
            self.assignment_type = assignment_type

        self.directory = directory
        self.assignments = assignments
        self.rerun = rerun
        return

    def simulate_parameter_list(self, parameter_list=None, processes=None):
        '''Run simulations for a list of parameters.

        Args:

        parameter_list: List of SystemParameters for which to run simulations.

        processes: The number of parellel processes to run.

        '''

        assert isinstance(parameter_list, list)
        logging.info('Running simulations for %d parameters in directory %s.',
                     len(parameter_list), self.directory)

        result = SimulatorResult(self, parameter_list, self.directory)
        if self.rerun:
            indices = list(range(len(parameter_list)))
        else:
            indices = [i for i in range(len(parameter_list))
                       if result.dataframes[i] is None]

        # Run simulations for any parameters without results.
        with ProcessPoolExecutor(max_workers=processes) as executor:
            dataframes = executor.map(self.simulate, [parameter_list[i] for i in indices])

        # Add the dataframes to the result
        for i, dataframe in zip(indices, dataframes):
            result.append_results(i, dataframe)

        return result

    def simulate(self, parameters):
        '''Run simulations for a single set of parameters.

        Writes the results to disk as .csv files. Will also write the
        best assignment matrix found to disk in the same directory.

        Args:

        parameters: SystemParameters for which to run simulations.

        '''

        assert isinstance(parameters, SystemParameters)
        assert os.path.exists(self.directory)
        logging.debug('Running simulations for %s: %s',
                      self.directory, parameters.identifier())

        best_assignment = None
        best_avg_load = math.inf
        best_avg_delay = math.inf

        printout_interval = datetime.timedelta(seconds=10)
        last_printout = datetime.datetime.utcnow()

        results = list()
        for i in range(self.assignments):

            # Print progress periodically
            if datetime.datetime.utcnow() - last_printout > printout_interval:
                last_printout = datetime.datetime.utcnow()
                logging.info('%s in %s %f percent finished.', self.directory,
                             parameters.identifier(), i / self.assignments * 100)

            # Use parameter_eval if there is no solver.
            if self.solver is None:
                result = self.parameter_eval(parameters)
                if isinstance(result, dict):
                    result = pd.DataFrame(result)

            # Otherwise find an assignment and evaluate it.
            else:
                # Create an assignment
                assignment = self.solver.solve(
                    parameters,
                    assignment_type=self.assignment_type
                )
                if not assignment.is_valid():
                    logging.error('Assignment invalid for parameters: %s.', str(parameters))
                    return pd.DataFrame()

                # Evaluate it
                result = self.assignment_eval.evaluate(parameters, assignment)
                if isinstance(result, dict):
                    result = pd.DataFrame(result)

                # Keep the best assignment
                # if result['delay'].mean() < best_avg_delay:
                #     best_assignment = assignment
                #     best_avg_delay = result['delay']

            # Add the assignment index and append
            result['assignment'] = i * np.ones(len(result))
            results.append(result)

        # Concatenate DataFrames and write to disk
        dataframe = pd.concat(results)
        filename = os.path.join(self.directory, parameters.identifier() + '.csv')
        dataframe.to_csv(filename)

        # Write the best assignment to disk
        if best_assignment is not None:
            best_assignment.save(directory=self.directory)

        return dataframe
