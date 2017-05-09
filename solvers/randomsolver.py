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

'''This solver creates a random assignment'''

import random
import model
from assignments import Assignment
from assignments.sparse import SparseAssignment
from solvers import Solver

class RandomSolver(Solver):
    '''Create an assignment matrix randomly.'''

    def __init__(self):
        return

    def solve(self, par, assignment_type=None, optimized=False):
        '''Create an assignment matrix randomly.

        Args:

        par: System parameters

        assignment_type: Assignment kind. Defaults to SparseAssignment
        if set to None.

        optimized: If True, the solver will first assign as many rows
        as possible to all elements of the assignment matrix, and then
        assign any remaining rows randomly. Defaults to False.

        Returns: The resulting assignment object.

        '''

        assert isinstance(par, model.SystemParameters)
        if optimized:
            rows_per_element = int(par.num_coded_rows / (par.num_partitions * par.num_batches))
        else:
            rows_per_element = 0

        if assignment_type is None:
            assignment = SparseAssignment(par, gamma=rows_per_element)
        else:
            assignment = assignment_type(par, gamma=rows_per_element)

        count_by_partition = [rows_per_element * par.num_batches] * par.num_partitions
        assignment = self.assign_remaining_random(par, assignment, count_by_partition)
        return assignment

    def assign_remaining_random(self, par, assignment, count_by_partition):
        '''Assign any remaining rows randomly.

        Args:
        par: System parameters

        assignment: Assignment object

        count_by_partition: A list of row counts by partition.

        '''

        assert len(count_by_partition) == par.num_partitions, \
            'count_by_partition must be of length equal to the number of partitions.'

        # Create a set containing the indices of the partially assigned
        # partitions
        coded_rows_per_partition = par.num_coded_rows / par.num_partitions
        partial_partitions = set()

        for partition in range(par.num_partitions):
            if count_by_partition[partition] < coded_rows_per_partition:
                partial_partitions.add(partition)

        # Assign symbols randomly row-wise
        rows = dict()
        for row in range(par.num_batches):
            cols = dict()
            row_sum = assignment.batch_union({row}).sum()
            while row_sum < par.rows_per_batch:
                col = random.sample(partial_partitions, 1)[0]
                if col in cols:
                    cols[col] += 1
                else:
                    cols[col] = 1

                # Increment the count
                count_by_partition[col] += 1
                row_sum += 1

                # Remove the partition index if there are no more assignments
                if count_by_partition[col] == coded_rows_per_partition:
                    partial_partitions.remove(col)

            rows[row] = cols

        for row, cols in rows.items():
            assignment = assignment.increment([row]*len(cols),
                                              list(cols.keys()),
                                              list(cols.values()))
        return assignment

    @property
    def identifier(self):
        '''Return a string identifier for this object.'''
        return self.__class__.__name__
