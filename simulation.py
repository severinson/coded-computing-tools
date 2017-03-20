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

import logging
import math
import os
from multiprocessing import Pool
import pandas as pd
import evaluation
import model

class Simulator(object):
    """ Create a simulator.

    Writes the results to disk as .csv files. Will also write the best
    assignment matrix found to disk in the same directory.
    """

    def __init__(self, solver=None, par_eval=None, directory=None,
                 num_assignments=1, num_samples=1000, rerun=False,
                 verbose=False):
        """Creates a assignment performance simulator.

        Writes the results to disk as .csv files. Will also write the best
        assignment matrix found to disk in the same directory.

        Args:

        solver: The solver to use when creating assignments. A
        par_eval method must be provided if set to None.

        par_eval: A method that takes a parameter object as its single
        argument and returns a dict with entries 'load' and 'delay'
        (and possibly others). This argument must be None if a solver
        is provided.

        directory: The directory to store the results in.

        num_assignments: The result is calculated over this number of
        generated assignments. Set this to a large number if
        assignments are generated randomly, and set it to 1 if
        assignments are deterministic.

        num_samples: If par_eval is set to None, assignment
        performance is evaluated by sampling its performance
        num_samples times and computing the average.

        rerun: If True, any result stored on disk are overwritten.

        verbose: Run the solver in verbose mode if True.

        """

        if solver is None:
            assert par_eval is not None
        elif solver is not None:
            assert par_eval is None

        if par_eval is not None:
            assert solver is None

        assert isinstance(directory, str)
        assert isinstance(num_assignments, int)
        assert isinstance(num_samples, int)
        assert isinstance(rerun, bool)
        assert isinstance(verbose, bool)

        # Make sure the directory name ends with a /
        if directory[-1] != '/':
            directory += '/'

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.solver = solver
        self.par_eval = par_eval
        self.directory = directory
        self.num_assignments = num_assignments
        self.num_samples = num_samples
        self.rerun = rerun
        self.verbose = verbose
        return

    def simulate_parameter_list(self, parameter_list=None, processes=4):
        """ Run simulations for a list of parameters.

        Args:
        parameter_list: List of SystemParameters for which to run simulations.
        processes: The number of parellel processes to run.
        """

        assert isinstance(parameter_list, list)
        logging.info('Running simulations for %d parameters.', len(parameter_list))

        # Run the simulations
        with Pool(processes=processes) as pool:
            _ = pool.map(self.simulate, parameter_list)

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

        result = evaluation.objective_function_sampled(par,
                                                       assignment.assignment_matrix,
                                                       assignment.labels,
                                                       num_samples=self.num_samples)
        return result

    def simulate(self, parameters):
        """Run simulations for a single set of parameters.

        Writes the results to disk as .csv files. Will also write the best
        assignment matrix found to disk in the same directory.

        Args:
        parameters: SystemParameters for which to run simulations.

        """

        assert isinstance(parameters, model.SystemParameters)
        assert os.path.exists(self.directory)
        par = parameters

        # Try to load the results from disk
        filename = self.directory + par.identifier() + '.csv'
        if os.path.isfile(filename) and not self.rerun:
            logging.debug('Found results for %s on disk. Skipping.', filename)
            return

        logging.debug('Running simulations for %s', filename)

        best_assignment = None
        best_avg_load = math.inf

        results = list()
        for i in range(self.num_assignments):
            if (i + 1) % 10 == 0:
                logging.info('%s %d\% finished.', filename, (i + 1) / self.num_assignments * 100)

            # If solver is None we should run the analysis provided by
            # the par_eval function.
            if self.solver is None:
                result = self.par_eval(par, num_samples=self.num_samples)

            # If a solver is provided we should find an assignment
            # using the solver and analyze its performance.
            else:
                # Create an assignment
                assignment = self.solver.solve(par, verbose=self.verbose)
                assert model.is_valid(par, assignment.assignment_matrix, verbose=self.verbose)

                # Evaluate it
                result = self.evaluate_assignment(par, assignment)

                # Keep the best assignment
                if result['load'] < best_avg_load:
                    best_assignment = assignment
                    best_avg_load = result['load']

            results.append(result)

        # Create a pandas dataframe and write it to disk
        dataframe = pd.DataFrame(results)
        dataframe.to_csv(self.directory + par.identifier() + '.csv')

        # Write the best assignment to disk
        if isinstance(best_assignment, model.Assignment):
            best_assignment.save(directory=self.directory)

        return
