'''tests of the overhead.py module'''

import unittest
import model
import overhead
import plot

def get_parameters():
    '''Get a list of parameters for the partitioning plot.'''
    rows_per_batch = 250
    num_servers = 9
    q = 6
    num_outputs = q
    server_storage = 1/3
    num_partitions = 250
    parameters = list()
    return model.SystemParameters(
        rows_per_batch=rows_per_batch,
        num_servers=num_servers,
        q=q,
        num_outputs=num_outputs,
        server_storage=server_storage,
        num_partitions=num_partitions,
    )

def get_parameters():
    '''Get a list of parameters for the N to n ratio plot.'''
    rows_per_batch = 200
    num_servers = 201
    q = 134
    num_partitions = rows_per_batch
    num_columns = None
    parameters = list()
    num_outputs = 100*q
    server_storage = 1/q
    parameters = model.SystemParameters(
        rows_per_batch=rows_per_batch,
        num_servers=num_servers,
        q=q,
        num_outputs=num_outputs,
        server_storage=server_storage,
        num_partitions=num_partitions,
        num_columns=num_columns,
    )
    return parameters

class Tests(unittest.TestCase):

    def test_batches_by_server(self):
        p = get_parameters()
        storage = overhead._batches_by_server(p.num_servers, p.muq)
        batches = [0 for _ in range(p.num_batches)]
        for s in storage:
            for i in s:
                batches[i] += 1
        for j in batches:
            self.assertEqual(j, p.muq)
        return

    def test_delay_from_order(self):
        p = get_parameters()
        order = list(range(p.num_servers))
        storage = overhead._batches_by_server(p.num_servers, p.muq)
        for overh in [1, 1.25, 1.3, 1.43, 1.44]:
            dct = overhead.delay_from_order(p, order, overh)
            servers = dct['servers']
            self.assertGreaterEqual(servers, p.q)
            self.assertLessEqual(servers, p.num_servers)
            batches = overhead._batches_from_order(storage, order[:servers])
            rows = overhead._rows_from_batches(p, batches)
            self.assertEqual(rows, len(batches)*p.rows_per_batch)
            self.assertGreaterEqual(rows, p.num_source_rows*overh)
            if servers > p.q:
                servers -= 1
                batches = overhead._batches_from_order(storage, order[:servers])
                rows = overhead._rows_from_batches(p, batches)
                self.assertLess(rows, p.num_source_rows*overh)

