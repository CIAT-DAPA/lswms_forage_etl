import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from datetime import datetime
import json
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.funcs import *

print ("")
print ("")
print ("         Converting point values to raster files ")
print ("")
print ("")
print ("")


# Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
outputs_path = os.path.join(project_root, 'outputs')
data_path = os.path.join(project_root, 'data')


layers_path = os.path.join(data_path, "layers")

biomass_path = os.path.join(layers_path, "biomass_et")


raster_output= os.path.join(outputs_path,  "raster.tif")

file_new_log_final = os.path.join(outputs_path,  "new_data_list_FINAL.txt")

results_output= os.path.join(outputs_path,  "results.csv")
if check_size(results_output):
    truncate_file(file_new_log_final)
    exit_program()

#get new files for Geoserver
What_new_files = []

def process_data(result_path):
    data = pd.read_csv(result_path)
    data['biom'] = round(data['biom'], 2)
    data['biom'] = abs(data['biom'])
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = data[['lon', 'lat', 'year', 'month', 'day', 'biom']]
    data['day'] = pd.to_numeric(data['day'])
    data['month'] = pd.to_numeric(data['month'])
    geodata = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']), crs='EPSG:4326')
    return geodata



geo_data =process_data(results_output);

raster = rasterio.open(raster_output, 'w', driver='GTiff', height=300, width=260, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(36, 15, 0.05, 0.05))

output_dir = biomass_path

def create_fresh_list_final():
        #write all new files

        try:
                os.remove(file_new_log_final)
        except OSError:
                pass

        with open(file_new_log_final, 'w') as the_file:
                for x in What_new_files:
                        the_file.write(x + '\n')

def create_raster_files(data,dir,r):
    for year in data['year'].unique():
        for month in data['month'].unique():
            for day in data['day'].unique():
                subset_data = data[(data['year'] == year) & (data['month'] == month) & (data['day'] == day)]
                if len(subset_data) > 0:
                    rasterized = rasterize([(geom, value) for geom, value in zip(subset_data.geometry, subset_data['biom'])], out_shape=r.shape, transform=r.transform, fill=0, all_touched=True)
                    rasterized[rasterized <= 0] = -99999
                    file_name = f"biomass_{year}{month:02d}{day:02d}.tif"
                    filename = f"{dir}/{file_name}"
                    What_new_files.append(file_name)
                    with rasterio.open(filename, 'w', driver='GTiff', height=rasterized.shape[0], width=rasterized.shape[1], count=1, dtype='float32', crs='EPSG:4326', transform=r.transform, nodata=-99999) as dst:
                        dst.write(rasterized, 1)
create_raster_files(geo_data,output_dir,raster)
create_fresh_list_final()                    