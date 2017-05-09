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

'''This module contains tests of the model module.

'''

import math
import unittest
import model

class Modeltests(unittest.TestCase):
    '''Tests for the model'''
    def test_multicast_set_size_1(self):
        '''Verify that multicast set sizes are computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=6, q=4, num_outputs=4,
                                            server_storage=1/2, num_partitions=5)
        self.assertEqual(parameters.multicast_set_size_1(), 2)

        # Strategy 2 isn't available for these parameters
        with self.assertRaises(model.ModelError):
            parameters.multicast_set_size_2()

        return

    def test_multicast_set_size_2(self):
        '''Verify that multicast set sizes are computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=10, q=9, num_outputs=9,
                                            server_storage=1/3, num_partitions=5)

        self.assertEqual(parameters.multicast_set_size_1(), 3)
        self.assertEqual(parameters.multicast_set_size_2(), 2)
        return

    def test_multicast_load_1(self):
        '''Verify that the multicast load is computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=2, num_servers=6, q=4, num_outputs=4,
                                            server_storage=1/2, num_partitions=5)
        load_1, load_2 = parameters.multicast_load()
        self.assertEqual(load_1 * parameters.num_source_rows, 12)
        self.assertEqual(load_2, math.inf)
        return

    def test_multicast_load_2(self):
        '''Verify that the multicast load is computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=10, q=9, num_outputs=9,
                                            server_storage=1/3, num_partitions=5)
        load_1, load_2 = parameters.multicast_load()
        self.assertAlmostEqual(load_1 * parameters.num_source_rows, 840)
        self.assertEqual(load_2 * parameters.num_source_rows, 2100)
        return

    def test_unpartitioned_load_1(self):
        '''Verify that the unpartitioned load is computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=6, q=4, num_outputs=4,
                                            server_storage=1/2, num_partitions=5)
        load_1 = parameters.unpartitioned_load(strategy='1')
        with self.assertRaises(model.ModelError):
            load_2 = parameters.unpartitioned_load(strategy='2')

        load_best = parameters.unpartitioned_load(strategy='best')
        self.assertEqual(load_1, load_best)
        self.assertAlmostEqual(load_1, 1.4)
        return

    def test_unpartitioned_load_2(self):
        '''Verify that the unpartitioned load is computed correctly.'''
        parameters = model.SystemParameters(rows_per_batch=5, num_servers=10, q=9, num_outputs=9,
                                            server_storage=1/3, num_partitions=5)
        load_1 = parameters.unpartitioned_load(strategy='1')
        load_2 = parameters.unpartitioned_load(strategy='2')
        load_best = parameters.unpartitioned_load(strategy='best')
        self.assertEqual(load_1, load_best)
        self.assertAlmostEqual(load_1, 2.8888888888888897)
        return
