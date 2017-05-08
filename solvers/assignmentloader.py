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

'''This module emulates an assignment solver, but actually loades an
assignment matrix from disk and hands it back.

'''

import logging
import model
from assignments.cached import CachedAssignment

class AssignmentLoader(object):
    '''This module emulates an assignment solver, but actually loades an
    assignment matrix from disk and hands it back.

    '''
    def __init__(self, directory):
        '''Create an assignment loader.

        Args:

        directory: Directory to load the assignment from.

        '''

        assert isinstance(directory, str)
        self.directory = directory
        return

    def solve(self, parameters):
        '''Load assignment from disk.

        Args:

        par: System parameters

        Returns: The loaded assignment

        '''
        assert isinstance(parameters, model.SystemParameters)
        logging.debug('Loading assignment from %s.', self.directory)
        return CachedAssignment.load(parameters, directory=self.directory)

    @property
    def identifier(self):
        """ Return a string identifier for this object. """
        return self.__class__.__name__
