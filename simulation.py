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
import pandas as pd
import evaluation
import model

def simulate(parameters=None, solver=None, directory=None, num_runs=1,
             num_samples=100, verbose=False):

    """ Run simulations for the parameters returned by get_parameters()

    Writes the results to disk as .csv files. Will also write the best
    assignment matrix found to disk in the same directory.

    Args:
    parameters: List of SystemParameters for which to run simulations.
    solver: The solver to use. A function pointer.
    directory: The directory to store the results in.
    num_runs: The number of simulations to run.
    num_samples: The number of objective function samples.
    verbose: Passed to the solver.
    """

    assert solver is not None
    assert isinstance(directory, str)
    assert isinstance(num_runs, int)
    assert isinstance(num_samples, int)
    assert isinstance(verbose, bool)

    # Add the solver identifier to the directory name
    directory += solver.identifier + '/'

    # Create the directory to store the results in if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for par, i in zip(parameters, range(1, len(parameters) + 1)):
        if not isinstance(par, model.SystemParameters):
            warnings.warn('Attempted to run simulation for something not of type SystemParameters:', par)

        # Try to load the results from disk
        try:
            pd.read_csv(directory + par.identifier() + '.csv')
            print('Found results for', directory, par.identifier(),
                  'on disk. Skipping.',
                  '(' + str(i) + '/' + str(len(parameters)) + ')')
            continue

        # Run the simulations if we couldn't find any
        except FileNotFoundError:
            print('Running simulations for', directory, par.identifier(),
                  '(' + str(i) + '/' + str(len(parameters)) + ')')

            best_assignment = None
            best_avg_load = math.inf

            results = list()
            for _ in range(num_runs):

                # Create an assignment
                assignment = solver.solve(par, verbose=verbose)
                assert model.is_valid(par, assignment.assignment_matrix, verbose=True)

                # Evaluate it
                avg_load, avg_delay = evaluation.objective_function_sampled(par,
                                                                            assignment.assignment_matrix,
                                                                            assignment.labels,
                                                                            num_samples=num_samples)

                # Store the results
                result = dict()
                result['load'] = avg_load
                result['delay'] = avg_delay
                results.append(result)

                # Keep the best assignment
                if avg_load < best_avg_load:
                    best_assignment = assignment
                    best_avg_load = avg_load

                # Write a dot to show that progress is being made
                sys.stdout.write('.')
                sys.stdout.flush()

            print('')

            # Create a pandas dataframe and write it to disk
            dataframe = pd.DataFrame(results)
            dataframe.to_csv(directory + par.identifier() + '.csv')

            # Write the best assignment to disk
            if isinstance(best_assignment, model.Assignment):
                best_assignment.save(directory=directory)

    return
