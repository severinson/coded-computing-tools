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
from multiprocessing import Pool
import numpy as np
import pandas as pd
from solvers import Solver
from model import SystemParameters
from assignments.sparse import SparseAssignment
from evaluation import AssignmentEvaluator, ParameterEvaluator

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

        parameter_eval: ParameterEvaluator object. Must be None if
        solver or assignment_eval is provided.

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
            assert isinstance(parameter_eval, ParameterEvaluator)
            assert solver is None
            assert assignment_eval is None

        if assignment_eval is None:
            assert isinstance(parameter_eval, ParameterEvaluator)
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

    def simulate_parameter_list(self, parameter_list=None, processes=4):
        '''Run simulations for a list of parameters.

        Args:

        parameter_list: List of SystemParameters for which to run simulations.

        processes: The number of parellel processes to run.

        '''

        assert isinstance(parameter_list, list)
        logging.info('Running simulations for %d parameters in directory %s..',
                     len(parameter_list), self.directory)

        # Run the simulations
        with Pool(processes=processes) as pool:
            _ = pool.map(self.simulate, parameter_list)

        return

    def simulate(self, parameters):
        '''Run simulations for a single set of parameters.

        Writes the results to disk as .csv files. Will also write the
        best assignment matrix found to disk in the same directory.

        Args:

        parameters: SystemParameters for which to run simulations.

        '''

        assert isinstance(parameters, SystemParameters)
        assert os.path.exists(self.directory)

        # Try to load the results from disk
        filename = os.path.join(self.directory, parameters.identifier() + '.csv')
        if os.path.isfile(filename) and not self.rerun:
            logging.debug('Found results for %s on disk. Skipping.', filename)
            return

        logging.debug('Running simulations for %s', filename)

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
                logging.info('%s %f percent finished.', filename, i / self.assignments * 100)

            # Use parameter_eval if there is no solver.
            if self.solver is None:
                result = self.parameter_eval.evaluate(parameters)
                if isinstance(result, dict):
                    result = pd.DataFrame(result)

            # Otherwise find an assignment and evaluate it.
            else:
                # Create an assignment
                assignment = self.solver.solve(parameters, assignment_type=self.assignment_type)
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
        dataframe.to_csv(filename)

        # Write the best assignment to disk
        if best_assignment is not None:
            best_assignment.save(directory=self.directory)

        return dataframe
