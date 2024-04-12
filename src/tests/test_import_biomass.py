import unittest
from unittest.mock import MagicMock
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.tools import GeoserverClient


class TestImportBiomass(unittest.TestCase):

    def setUp(self):
        # Set up test data and parameters
        self.geo_url = 'test_geoserver_url'
        self.geo_user = 'test_geoserver_user'
        self.geo_pwd = 'test_geoserver_pwd'
        self.workspace_name = 'test_workspace'
        self.forecast_type = 'test_forecast'
        self.folder_data = 'test_data_folder'
        self.outputs_path = 'test_outputs_path'
        self.folder_layers = os.path.join(self.folder_data, "layers")
        self.folder_properties = os.path.join(self.folder_layers, self.forecast_type+"_properties")
        self.folder_tmp = os.path.join(self.folder_data, "tmp")
        self.fname = os.path.join(self.outputs_path, "new_data_list_FINAL.txt")

        with open(self.fname, 'w') as f:
            f.write("biomass_20240101.tif\n")

    def test_geoserver_interaction(self):
        # Mock GeoserverClient
        geoclient = GeoserverClient(self.geo_url, self.geo_user, self.geo_pwd)
        geoclient.connect = MagicMock()
        geoclient.get_workspace = MagicMock(return_value=self.workspace_name)
        geoclient.get_store = MagicMock(return_value=None)
        geoclient.create_mosaic = MagicMock()
        geoclient.update_mosaic = MagicMock()

        # Call the script function
        import_script_function(self.geo_url, self.geo_user, self.geo_pwd, self.workspace_name, self.forecast_type, self.folder_data, self.outputs_path)

        # Assertions
        geoclient.connect.assert_called_once()
        geoclient.get_workspace.assert_called_once_with(self.workspace_name)
        geoclient.get_store.assert_called_once_with(self.forecast_type)
        geoclient.create_mosaic.assert_called_once()
        geoclient.update_mosaic.assert_not_called()  # Assuming the store is not found, so only create_mosaic should be called

    def tearDown(self):
        # Clean up temporary files or resources if any
        if os.path.exists(self.fname):
            os.remove(self.fname)

if __name__ == '__main__':
    unittest.main()
