import json
from glob import glob
import h5py as h5
import numpy as np
import sys
import os
import shutil
import BiomassAggregate
import BiomassForecast
import BiomassHindcasts
from datetime import datetime
# suppress future warning
import warnings
from shutil import copyfile

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')

with open(filepath) as f:
    data = json.load(f)


inputs_path = os.path.join(project_root, 'inputs')
outputs_path = os.path.join(project_root, 'outputs')

shapefile_folder = os.path.join(inputs_path, "Shapefiles")



def main():
    
    shape_files = [
        os.path.join(shapefile_folder, "woredas.shp"),
    ]

    shape_file_columns = [
        'ADM3_PCODE'
    ]

    create_new_time_series_list = [
        True
    ]

   

    convert_BIOMASS_from_scratch_list = [
        True
    ]
    

    for shape, column, new_time_series, from_scratch in zip(shape_files, shape_file_columns,
                                                            create_new_time_series_list,
                                                            convert_BIOMASS_from_scratch_list):

        create_new_time_series = new_time_series

        convert_BIOMASS_from_scratch = from_scratch


        shapefile_filepath = shape

        data_path = os.path.join(project_root, 'data')
        layers_path = os.path.join(data_path, "layers")
        biomass_path = os.path.join(layers_path, "biomass_et")

        name_of_shapefile_column = column
        
        smoothed_biomass = sorted(glob(os.path.join(biomass_path, "*.tif")))


        if create_new_time_series and convert_BIOMASS_from_scratch:

            create_time_series = BiomassAggregate.aggregate_time_series(smoothed_biomass,shapefile_filepath,create_new_time_series,name_of_shapefile_column,base_all_touched=False,
                                                buffer_tiny_polygons=True,
                                                debug=True)
            
            create_time_series.open_files()

           
            database_name = os.path.splitext(os.path.basename(shapefile_filepath))[0]

            create_hindcasts = BiomassHindcasts.hindcast(database_name)
           
            create_hindcasts.open_dataset()

            create_the_forecasts = BiomassForecast.forecast(database_name,
                                                     shapefile_filepath,
                                                     name_of_shapefile_column)
            create_the_forecasts.open_dataset()
        

        

        elif create_new_time_series is True and convert_BIOMASS_from_scratch is False:

            create_time_series = \
                BiomassAggregate.aggregate_time_series(smoothed_biomass,
                                                shapefile_filepath,
                                                create_new_time_series,
                                                name_of_shapefile_column,
                                                base_all_touched=False,
                                                buffer_tiny_polygons=True,
                                                debug=True)

            
            create_time_series.open_files()

            database_name = os.path.splitext(os.path.basename(shapefile_filepath))[0]
            create_hindcasts = BiomassHindcasts.hindcast(database_name)
            create_hindcasts.open_dataset()

            create_the_forecasts = BiomassForecast.forecast(database_name,
                                                     shapefile_filepath,
                                                     name_of_shapefile_column)

            create_the_forecasts.open_dataset()
            
                
        shutil.rmtree(
            os.path.join(outputs_path, "Output_check"))

        os.mkdir(
            os.path.join(outputs_path, "Output_check"))


if __name__ == "__main__":
    main()
