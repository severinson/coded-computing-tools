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

'''This package contains functions used to evaluate the performance
assignments, or of paramaters. This file contains abstract base
classes for both kinds.

'''

from abc import ABC, abstractmethod

class EvaluationError(Exception):
    '''Base class for exceptions thrown by this package.'''

class AssignmentEvaluator(ABC):
    '''Assignment evaluator abstract base class.

    '''

    @abstractmethod
    def evaluate(parameters, assignment, num_samples=1000):
        '''Evaluate the communication load and computational delay of an
        assignment.

        Args:

        parameters: System parameters.

        assignment: Assignment object.

        num_samples: Number of samples to take.

        Returns: A Pandas dataframe of length num_samples with
        performance samples.

        '''

class ParameterEvaluator(ABC):
    '''Parameter evaluator abstract base class.

    '''

    @abstractmethod
    def evaluate(parameters):
        '''Evaluate the communication load and computational delay of a set of
        a set of parameters.

        Args:

        parameters: System parameters.

        Returns: A Pandas dataframe of length 1 with the results.

        '''
