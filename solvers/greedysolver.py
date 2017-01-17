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

""" Greedy assignment solver.

Finds an assignment by assigning one row at a time, always greedily
selecting the row with largest improvement.
"""

import model

class GreedySolver(object):
    """ Greedy assignment solver.

    Finds an assignment by assigning one row at a time, always greedily
    selecting the row with largest improvement.
    """

    def __init__(self):
        return

    def solve(self, par, verbose=False):
        """ Greedy assignment solver.

        Finds an assignment by assigning one row at a time, always greedily
        selecting the row with largest improvement.

        Args:
        par: System parameters
        verbose: Print extra messages if True

        Returns:
        The resulting assignment
        """

        assignment = model.Assignment(par)

        # Separate symbols by partition
        symbols_per_partition = par.num_coded_rows / par.num_partitions
        symbols_separated = [symbols_per_partition for x in range(par.num_partitions)]

        # Assign symbols row by row
        for row in range(par.num_batches):
            if verbose:
                print('Assigning batch', row, 'Bound:', assignment.bound())

            # Assign rows_per_batch rows per batch
            for _ in range(par.rows_per_batch):
                score_best = 0
                best_col = 0

                # Try one column at a time
                for col in range(par.num_partitions):

                    # If there are no more symbols left for this column to
                    # assign, continue
                    if symbols_separated[col] == 0:
                        continue

                    # Evaluate the score of the assignment
                    score_updated = assignment.evaluate(row, col)
                    score_updated = score_updated + symbols_separated[col] / symbols_per_partition

                    # If it's better, store it
                    if score_updated > score_best:
                        score_best = score_updated
                        best_col = col

                # Store the best assignment from this iteration
                assignment = assignment.increment(row, best_col)
                symbols_separated[best_col] = symbols_separated[best_col] - 1

        return assignment

    @property
    def identifier(self):
        """ Return a string identifier for this object. """
        return self.__class__.__name__
