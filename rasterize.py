import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from datetime import datetime
import json
import os
import numpy as np


# Constants
filepath = os.path.join(os.getcwd(), "data.json")

with open(filepath) as f:
    data = json.load(f)

all_path = data["base_directory"]

outputs_path = os.path.join(all_path, "outputs")

data_path = os.path.join(all_path, "data")

layers_path = os.path.join(data_path, "layers")

biomass_path = os.path.join(layers_path, "biomass_et")


raster_output= os.path.join(outputs_path,  "raster.tif")

results_output= os.path.join(outputs_path,  "results.csv")

data = pd.read_csv(results_output)

data['biom'] = round(data['biom'], 2)
data['pred'] = round(data['pred'], 2)
data['biom'] = abs(data['biom'])
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data[['lon', 'lat', 'year', 'month', 'day', 'biom']]
data['day'] = pd.to_numeric(data['day'])
data['month'] = pd.to_numeric(data['month'])

data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']), crs='EPSG:4326')

r = rasterio.open(raster_output, 'w', driver='GTiff', height=300, width=260, count=1, dtype='float32', crs='EPSG:4326', transform=from_origin(36, 15, 0.05, 0.05))

dir = biomass_path

for year in data['year'].unique():
    for month in data['month'].unique():
        for day in data['day'].unique():
            subset_data = data[(data['year'] == year) & (data['month'] == month) & (data['day'] == day)]
            
            if len(subset_data) > 0:
        
                rasterized = rasterize([(geom, value) for geom, value in zip(subset_data.geometry, subset_data['biom'])], out_shape=r.shape, transform=r.transform, fill=0, all_touched=True)
                rasterized[rasterized <= 0] = -99999
                
                filename = f"{dir}/biomass_{year}{month:02d}{day:02d}.tif"
                
                
                with rasterio.open(filename, 'w', driver='GTiff', height=rasterized.shape[0], width=rasterized.shape[1], count=1, dtype='float32', crs='EPSG:4326', transform=r.transform, nodata=-99999) as dst:
                    dst.write(rasterized, 1)