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

""" This solver creates a random assignment. """

import random
import model

class RandomSolver(object):
    """ Create an assignment matrix randomly. """

    def __init__(self):
        return

    def solve(self, par, verbose=False):
        """ Create an assignment matrix randomly.

        Args:
        par: System parameters
        verbose: Unused

        Returns:
        The resulting assignment object.
        """

        assert isinstance(par, model.SystemParameters)

        assignment = model.Assignment(par, score=False, index=False)
        assignment_matrix = assignment.assignment_matrix
        rows_by_partition = [0 for x in range(par.num_partitions)]
        self.assign_remaining_random(par, assignment_matrix, rows_by_partition)
        return assignment

    def assign_remaining_random(self, par, assignment_matrix, rows_by_partition):
        """ Assign any remaining rows randomly.

        Args:
        par: System parameters
        assignment_matrix: Assignment matrix
        rows_by_partition: A list of row counts by partition.
        """

        assert len(rows_by_partition) == par.num_partitions, \
            'rows_by_partition must be of length equal to the number of partitions.'

        # Create a set containing the indices of the partially assigned
        # partitions
        coded_rows_per_partition = par.num_coded_rows / par.num_partitions
        partial_partitions = set()

        for partition in range(par.num_partitions):
            if rows_by_partition[partition] < coded_rows_per_partition:
                partial_partitions.add(partition)

        # Assign symbols randomly row-wise
        for row in range(par.num_batches):
            row_sum = assignment_matrix[row].sum()
            while row_sum < par.rows_per_batch:

                # Get a random column index corresponding to a partial partition
                col = random.sample(partial_partitions, 1)[0]

                # Increment the count
                assignment_matrix[row, col] += 1
                rows_by_partition[col] += 1
                row_sum += 1

                # Remove the partition index if there are no more assignments
                if rows_by_partition[col] == coded_rows_per_partition:
                    partial_partitions.remove(col)
        return

    @property
    def identifier(self):
        """ Return a string identifier for this object. """
        return self.__class__.__name__
