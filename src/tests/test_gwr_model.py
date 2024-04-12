import unittest
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from mgwr.gwr import GWR
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.gwr_model import *

class TestGwrModel(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with test data
        self.test_data = pd.DataFrame({'lon': [1, 2, 3], 'lat': [4, 5, 6], 'ndvi': [0.1, 0.2, 0.3],
                                       'sm': [0.4, 0.5, 0.6], 'preci': [0.7, 0.8, 0.9]})
        self.test_data.to_csv('test_data.csv', index=False)

    def tearDown(self):
        # Clean up temporary files after each test
        os.remove('test_data.csv')

    def test_process_data(self):
        # Test loading and processing data from a CSV file
        processed_data = process_data('test_data.csv')
        self.assertIsInstance(processed_data, GeoDataFrame)
        self.assertEqual(len(processed_data), 3)  # Assuming 3 rows of test data

    def test_run_gwr_model(self):
        # Test running GWR model and generating predictions
        processed_data = process_data('test_data.csv')
        results = run_gwr_model(processed_data)
        self.assertIsInstance(results, GWR)
        self.assertEqual(len(results.predictions), 3)  # Assuming 3 rows of test data

    def test_output_results(self):
        # Test writing results to a CSV file
        output_path = 'test_results.csv'
        processed_data = process_data('test_data.csv')
        results = run_gwr_model(processed_data)
        output_path = output_results(results, output_path)
        self.assertTrue(os.path.isfile(output_path))

if __name__ == '__main__':
    unittest.main()