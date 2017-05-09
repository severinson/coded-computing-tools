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

'''Solvers create assignment matrices. This file contains the solver
abstract base class.

'''

from abc import ABC, abstractmethod

class SolverError(Exception):
    '''Base class for exceptions thrown by this package.'''

class Solver(ABC):
    '''Solver abstract base class.

    '''
    @abstractmethod
    def solve(self, parameters, assignment_type=None):
        '''Create an assignment.

        Args:

        parameters: System parameters.

        assignment_type: Type of assignment matrix to return.

        Returns: The resulting assignment.

        '''

    @property
    def identifier(self):
        '''Return a string identifier for this object.'''
        return self.__class__.__name__
