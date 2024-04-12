import unittest
import sys,os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.rasterize import *

class TestRasterize(unittest.TestCase):

    def setUp(self):
        # Create a temporary results CSV file with test data
        self.test_data = pd.DataFrame({'lon': [39.41, 39.445, 39.492], 'lat': [3.271, 3.261, 3.254], 'date': [20240101,20240101,20240101],'year': [2024, 2024, 2024],
                                       'month': [1, 1, 1], 'day': [1, 1, 1],
                                        'biom': [1.79040380360565, 1.78896823494216, 1.79198639098456]})
        self.test_data.to_csv('test_results.csv', index=False)

    def tearDown(self):
        # Clean up temporary files after each test
        os.remove('test_results.csv')
        #for file in os.listdir('temp_raster_files'):
            #os.remove(os.path.join('temp_raster_files', file))

    def test_process_data(self):
        # Test if process_data function correctly converts CSV to GeoDataFrame
        self.test_data = process_data('test_results.csv')
        self.assertIsInstance(self.test_data, gpd.GeoDataFrame)
        self.assertEqual(len(self.test_data), 3)  # Assuming 3 rows of test data

    def test_create_raster_files(self):
        # Test if create_raster_files function correctly generates raster files
        raster = rasterio.open('test_raster.tif', 'w', driver='GTiff', height=300, width=260, count=1,dtype='float32', crs='EPSG:4326', transform=rasterio.transform.from_origin(36, 15, 0.05, 0.05))
        create_raster_files(self.test_data, 'temp_raster_files', raster)
        raster.close()
        raster_files = os.listdir('temp_raster_files')
        self.assertEqual(len(raster_files), 1)  # Assuming only one raster file is created
        self.assertTrue(raster_files[0].startswith('biomass_20240101'))  # Assuming the file name follows a certain pattern

    def test_create_fresh_list_final(self):
        # Test if create_fresh_list_final function correctly creates or updates the output file
        create_fresh_list_final()
        self.assertTrue(os.path.exists('new_data_list_FINAL.txt'))  # Assuming the output file is created or updated

if __name__ == '__main__':
    unittest.main()
