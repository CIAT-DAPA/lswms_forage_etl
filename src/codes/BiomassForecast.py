import os
import sys
import json
import h5py as h5
import numpy as np
from datetime import datetime, timedelta
import GaussianProcesses
from glob import glob
import requests


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')

with open(filepath) as f:
    data = json.load(f)


inputs_path = os.path.join(project_root, 'inputs')
outputs_path = os.path.join(project_root, 'outputs')
forecasts_url = data["forecasts_url"]
api_key = data["api_key"]


class forecast:
    

    def __init__(self, database_path, shapefile_path, name_of_shapefile_column):
        self.database = database_path
        self.file = None
        self.shapefile_path = shapefile_path
        self.column_name = name_of_shapefile_column
        self.forecasts_url = forecasts_url
        self.api_key=api_key

    def raw_to_datetime(self, unformatted_date):
       
        return datetime.strptime(str(int(unformatted_date)), '%Y%m%d')

    def create_forecast(self):
       
        raw_to_datetime_Vec = np.vectorize(self.raw_to_datetime)

        dataset_names = list(self.file.keys())

        full_forecast_results = np.empty((len(dataset_names), 4))

        predicted_dates = []

        for dataset_no, dataset in enumerate(dataset_names):

            

            dataset_array = np.array(self.file[dataset], dtype=float)

            zero_mask = np.isnan(dataset_array[:-16, 0])

            dates = raw_to_datetime_Vec(dataset_array[:-16, 0])

            days = np.array([(date - dates[0]).days for date in dates])

            nan_mask = np.isnan(dataset_array[:-16, 1])

            biomass = dataset_array[:-16, 1][~nan_mask]

            days = days[~nan_mask]

            dates = dates[~nan_mask]

            predicted_days, predicted_biomass = GaussianProcesses.forecast(days, biomass)

            predicted_dates = [dates[0] + timedelta(days=day) for day in predicted_days]

        

            full_forecast_results[dataset_no, :] = predicted_biomass[-4:]

        
        if predicted_dates:
            full_forecast_dates = predicted_dates[-4:]
        else:
            full_forecast_dates = []
           

        
        return full_forecast_dates,full_forecast_results, dataset_names

    def open_dataset(self):
        
        
        self.file = h5.File(
            (os.path.join(outputs_path, "Databases", self.database+".h5")), 'r+')
        full_forecast_dates,full_hindcast_results, dataset_names = self.create_forecast()
        self.file.close()

        self.update_forecasts(full_forecast_dates, full_hindcast_results, dataset_names);

        return full_forecast_dates,full_hindcast_results, dataset_names


    
    def update_forecasts(self, full_forecast_dates, full_forecast_data, dataset_names):
        """
        Updates forecasts by sending all data in a single POST request with authentication.
        
        Parameters:
            full_forecast_dates (list): List of forecast dates.
            full_forecast_data (ndarray): Array of forecast data (means) for each dataset and date.
            dataset_names (list): List of dataset (woreda) names.
            api_key (str): The API key for authentication.
        """
        # Prepare data to send in a single POST request
        all_forecast_data = []

        for dataset_index, dataset_name in enumerate(dataset_names):
            for date_index, forecast_date in enumerate(full_forecast_dates):
                data_to_send = {
                    "extId": dataset_name,
                    "mean": round(full_forecast_data[dataset_index, date_index], 4),
                    "date": forecast_date.strftime("%Y-%m-%d")
                }
                all_forecast_data.append(data_to_send)

        # Define headers with the API key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Send the data to the API
        try:
            response = requests.post(self.forecasts_url, json=all_forecast_data, headers=headers)
            if response.status_code == 201:
                print("Forecast data saved successfully via API.")
            elif response.status_code == 400:
                print(f"Validation error: {response.json()}")
            elif response.status_code == 401:
                print("Authentication failed: Invalid or missing API key.")
            else:
                print(f"Unexpected response ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"Error sending forecast data API: {e}")

   