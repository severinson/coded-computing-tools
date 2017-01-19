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

""" Run simulations for the parameters returned by get_parameters()

Writes the results to disk as .csv files. Will also write the best
assignment matrix found to disk in the same directory.
"""

import math
import sys
import os
import warnings
from multiprocessing import Pool
from multiprocessing import TimeoutError
import pandas as pd
import evaluation
import model

class Simulator(object):
    """ Create a simulator.

    Writes the results to disk as .csv files. Will also write the best
    assignment matrix found to disk in the same directory.
    """

    def __init__(self, solver=None, directory=None, num_runs=1,
                 num_samples=1000, verbose=False):

        """ Create a simulator.

        Writes the results to disk as .csv files. Will also write the best
        assignment matrix found to disk in the same directory.

        Args:
        solver: The solver to use.
        directory: The directory to store the results in.
        num_runs: The number of simulations to run.
        num_samples: The number of objective function samples.
        verbose: Run the solverin verbose mode if True.
        """

        assert solver is not None
        assert isinstance(directory, str)
        assert isinstance(num_runs, int)
        assert isinstance(num_samples, int)
        assert isinstance(verbose, bool)

        # Create the directory to store the results in if it doesn't exist
        if not os.path.exists(directory + solver.identifier + '/'):
            os.makedirs(directory + solver.identifier + '/')

        self.solver = solver
        self.directory = directory
        self.num_runs = num_runs
        self.num_samples = num_samples
        self.verbose = verbose

        return

    def simulate_parameter_list(self, parameter_list=None, processes=4):
        """ Run simulations for a list of parameters.

        Args:
        parameter_list: List of SystemParameters for which to run simulations.
        processes: The number of parellel processes to run.
        """

        assert isinstance(parameter_list, list)
        print('Running simulations for', len(parameter_list), 'parameters.')

        # Run the simulations
        with Pool(processes=processes) as pool:
            result = pool.map(self.simulate, parameter_list)

        return

    def evaluate_assignment(self, par, assignment):
        """ Evaluate an assignment.

        Args:
        par: System parameters
        assignment: Assignment object to evaluate

        Returns:
        A dict with the entries ['load'] and ['delay'] storing the average load
        and delay for the given assignment.
        """

        avg_load, avg_delay = evaluation.objective_function_sampled(par,
                                                                    assignment.assignment_matrix,
                                                                    assignment.labels,
                                                                    num_samples=self.num_samples)

        # Store the results
        result = dict()
        result['load'] = avg_load
        result['delay'] = avg_delay

        return result

    def simulate(self, parameters):
        """ Run simulations for a single set of parameters.

        Writes the results to disk as .csv files. Will also write the best
        assignment matrix found to disk in the same directory.

        Args:
        parameters: SystemParameters for which to run simulations.
        """

        assert isinstance(parameters, model.SystemParameters)

        solver = self.solver
        directory = self.directory
        num_runs = self.num_runs
        num_samples = self.num_samples
        verbose = self.verbose

        # Add the solver identifier to the directory name
        directory += solver.identifier + '/'

        # Create the directory to store the results in if it doesn't exist
        if not os.path.exists(directory):
            warnings.warn('Results directory ' + directory + ' doesn\'t exist. Returning.')
            return

        i = 0
        par = parameters

        # Try to load the results from disk
        try:
            pd.read_csv(directory + par.identifier() + '.csv')
            print('Found results for', directory, par.identifier(),
                  'on disk. Skipping.',)
            return

        # Run the simulations if we couldn't find any
        except FileNotFoundError:
            print('Running simulations for', directory, par.identifier())

            best_assignment = None
            best_avg_load = math.inf

            results = list()
            for _ in range(num_runs):

                # Create an assignment
                assignment = solver.solve(par, verbose=verbose)
                assert model.is_valid(par, assignment.assignment_matrix, verbose=True)

                # Evaluate it
                result = self.evaluate_assignment(par, assignment)
                results.append(result)

                # Keep the best assignment
                if result['load'] < best_avg_load:
                    best_assignment = assignment
                    best_avg_load = result['load']

            # Create a pandas dataframe and write it to disk
            dataframe = pd.DataFrame(results)
            dataframe.to_csv(directory + par.identifier() + '.csv')

            # Write the best assignment to disk
            if isinstance(best_assignment, model.Assignment):
                best_assignment.save(directory=directory)

        return
