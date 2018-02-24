'''Tests for the rateless.py module'''

import unittest
import tempfile
import subprocess
import pandas as pd
import model
import rateless

from os import path

class Tests(unittest.TestCase):

    def test_julia(self):
        '''test integration with the Julia simulator'''
        num_inputs = 1000
        mode = 100
        delta = 0.01
        overhead = 1.3
        with tempfile.TemporaryDirectory() as d:
            filename = path.join(d, "output.csv")
            subprocess.run([
                "julia", "rateless.jl",
                "--code", "LT",
                "--num_inputs", str(num_inputs),
                "--mode", str(mode),
                "--delta", str(delta),
                "--overhead", str(overhead-1),
                "--write", filename,
            ])
            df = pd.read_csv(filename)
        self.assertTrue(len(df))

    def test_evaluate(self):
        '''test the evaluate method'''
        p = model.SystemParameters(
            rows_per_batch=250,
            num_servers=9,
            q=6,
            num_outputs=6,
            server_storage=1/3,
            num_partitions=250,
        )
        target_overhead = 1.3
        target_failure_probability = 0.01
        dct = rateless.evaluate(
            p,
            target_overhead=target_overhead,
            target_failure_probability=target_failure_probability,
        )
        print(dct)

