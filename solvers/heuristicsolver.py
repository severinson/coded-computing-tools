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

""" This solver creates an assignment using a heuristic block-diagonal structure. """

import model

class HeuristicSolver(object):
    """ This solver creates an assignment using a heuristic block-diagonal structure. """

    def __init__(self):
        return

    def assign_block(self, par, assignment_matrix, row, min_col, max_col):
        """ Assign rows to all slots between min_col and max_col.

        This method won't work when max_col - min_col > par.rows_per_batch.

        Args:
        par: System parameters
        assignment_matrix: The assignment matrix
        row: The assignment matrix rows to assign rows to
        min_col: Start of block
        max_col: End of block

        Returns:
        The wrapped column index. If there are 10 columns and max_col is 12, 2 is returned.
        """

        assert (max_col - min_col) <= par.rows_per_batch

        if min_col == max_col:
            return max_col

        wrapped_col = max_col
        if max_col >= par.num_partitions:
            wrapped_col = max_col - par.num_partitions
            assignment_matrix[row, 0:wrapped_col] += 1

        max_col = min(par.num_partitions, max_col)
        assignment_matrix[row][min_col:max_col] += 1

        return wrapped_col

    def solve(self, par, verbose=False):
        """ Create an assignment using a block-diagonal structure.

        Args:
        par: System parametetrs
        verbose: Print extra messages if True.

        Returns: The resulting assignment
        """

        assignment = model.Assignment(par, score=False, index=False)
        assignment_matrix = assignment.assignment_matrix

        # If there are more coded rows than matrix elements, add that
        # number of rows to each element.
        rows_per_element = int(par.num_coded_rows / (par.num_partitions * par.num_batches))
        if rows_per_element > 0:
            for row in range(par.num_batches):
                for col in range(par.num_partitions):
                    assignment_matrix[row, col] = rows_per_element

        # Assign the remaining rows in a block-diagonal fashion
        remaining_rows_per_batch = par.rows_per_batch - rows_per_element * par.num_partitions
        min_col = 0
        for row in range(par.num_batches):
            max_col = min_col + remaining_rows_per_batch
            min_col = self.assign_block(par, assignment_matrix, row, min_col, max_col)

        return assignment

    @property
    def identifier(self):
        """ Return a string identifier for this object. """
        return self.__class__.__name__
