import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.data_extraction import *

class TestGetImage(unittest.TestCase):

    @patch('codes.data_extraction.ee.ImageCollection')
    def test_getImage(self, mock_ImageCollection):
        # Mock the ImageCollection
        mock_collection = MagicMock()
        mock_ImageCollection.return_value = mock_collection

        # Call the function
        result = getImage("MODIS/061/MOD13A2", "2024-01-01", "2024-01-31", "NDVI")

        # Assertions
        mock_ImageCollection.assert_called_once_with("MODIS/061/MOD13A2")
        mock_collection.filter.assert_called_once_with(
            ee.Filter.date("2024-01-01", "2024-01-31"))
        mock_collection.select.assert_called_once_with("NDVI")
        self.assertEqual(result, mock_collection)

    # Add more tests for other functions...

if __name__ == '__main__':
    unittest.main()