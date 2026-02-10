import json
import sys
import os
import h5py as h5
import numpy as np
from datetime import datetime, timedelta
import GaussianProcesses

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')

with open(filepath) as f:
    data = json.load(f)


inputs_path = os.path.join(project_root, 'inputs')
outputs_path = os.path.join(project_root, 'outputs')

class hindcast:
    

    def __init__(self, database_path):
        self.database = database_path
        self.file = None
        self.hindcast_results = None
        self.database_file = None
        self.data_halfway_point = None

    def raw_to_datetime(self, unformatted_date):
        return datetime.strptime(str(int(unformatted_date)), '%Y%m%d')

    def create_hindcast(self):
        raw_to_datetime_vec = np.vectorize(self.raw_to_datetime)
        dataset_names = list(self.file.keys())
        data_length = len(self.file[dataset_names[0]][:, 0])
        self.data_halfway_point = data_length // 2

       
        self.hindcast_results = np.full((data_length - self.data_halfway_point + 16, 4), np.nan)

        amount_of_hindcasts = min(100, data_length - self.data_halfway_point)
        skip_freq = 1

        for dataset_no, dataset in enumerate(dataset_names):
            self.database_file = self.file[dataset]
            for run_counter, hindcast_counter in enumerate(range(
                    self.data_halfway_point,
                    min(self.data_halfway_point + (amount_of_hindcasts * skip_freq), data_length),
                    skip_freq)):

                dataset_array = np.array(self.database_file, dtype=float)[:hindcast_counter]
                dataset = dataset.replace("?", "")

                dates = raw_to_datetime_vec(dataset_array[:, 0])
                days = np.array([(date - dates[0]).days for date in dates])
                nan_mask = np.isnan(dataset_array[:, 1]) 
                biomass = dataset_array[:, 1][~nan_mask]
                days = days[~nan_mask]
                dates = dates[~nan_mask]

                predicted_days, predicted_biomass = GaussianProcesses.forecast(days, biomass)

                predicted_dates = [dates[0] + timedelta(days=day) for day in predicted_days]

                for save_counter in range(0, min(4, self.hindcast_results.shape[0] - run_counter)):
                    try:
                        self.hindcast_results[run_counter + save_counter, save_counter] = predicted_biomass[-4 + save_counter]
                    except Exception as e:
                        print("Error was: ", str(e))

               
                if hindcast_counter % 40 == 0:
                    np.save((os.path.join(outputs_path, "Output_check", str(
                        hindcast_counter) + str(dataset) + " hindcast is done.npy")), np.arange(0, 16))
                    

            self.save_the_data()

    def save_the_data(self):
        required_rows = len(self.database_file) + 16
        required_columns = 6 

        # Resize the number of rows
        self.database_file.resize(required_rows, axis=0)
        
        # Resize the number of columns
        self.database_file.resize((required_rows, required_columns))
        
        self.database_file.attrs['Column_Names'] = [
            'Date', 'Biomass', '0 lag time', '16 day lag time', '32 day lag time', '48 day lag time'
        ]

        self.database_file[self.data_halfway_point:, 2:] = self.hindcast_results




    def open_dataset(self):
        self.file = h5.File((os.path.join(outputs_path, "Databases", str(self.database)+".h5")), 'a')

        self.create_hindcast()
        self.file.close()
