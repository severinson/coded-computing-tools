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

'''An assignment stores which partitions are stored in which batch,
and at what servers. This file contains the assignment abstract base
class.

'''

from abc import ABC, abstractmethod

class AssignmentError(Exception):
    '''Base class for exceptions thrown by this package.'''

class Assignment(ABC):
    '''Abstract base class representing an assignment of encoded rows into
    batches.

    '''
    @abstractmethod
    def increment(self, rows, cols, values):
        pass

    @abstractmethod
    def decrement(self, rows, cols, values):
        pass

    @abstractmethod
    def batch_union(self, batch_indices):
        pass

    @abstractmethod
    def save(self, directory='./saved_assignments/'):
        pass

    @abstractmethod
    def load(self, par, directory='./saved_assignments/'):
        pass
