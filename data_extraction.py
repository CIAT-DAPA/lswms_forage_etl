import os
import json
import ee
import numpy as np
import pandas as pd
import datetime
from datetime import date,timedelta




# Constants
filepath = os.path.join(os.getcwd(), "data.json")

with open(filepath) as f:
    data = json.load(f)

all_path = data["base_directory"]
private_key = os.path.join(all_path, "private_key.json")
inputs_path = os.path.join(all_path, "inputs")
outputs_path = os.path.join(all_path, "outputs")
data_path = os.path.join(all_path, "data")
layers_path = os.path.join(data_path, "layers")
biomass_path = os.path.join(layers_path, "biomass_et")
grid_points=os.path.join(inputs_path,  "grid_points.xlsx")
sm_output=os.path.join(outputs_path,  "sm.csv")
ndvi_output=os.path.join(outputs_path,  "ndvi.csv")
preci_output=os.path.join(outputs_path,  "preci.csv")
combined_output=os.path.join(outputs_path,  "combined.csv")

service_account = data["service_account"]
credentials = ee.ServiceAccountCredentials(service_account, private_key)
ee.Initialize(credentials)
layers = os.listdir(biomass_path)
if len(layers)>0:
  sorted_layers = sorted(layers, reverse=True)
  latest_image=sorted_layers[0][:-4][8:]
  date_format= '%Y%m%d';
  date_obj = datetime.datetime.strptime(latest_image, date_format)
  start_date_obj =date_obj.date() + timedelta(days=1)
  start_date_ =start_date_obj.strftime('%Y-%m-%d')

else:
  start_date_ = "2024-01-01";

end_date_ = date.today().strftime('%Y-%m-%d')
# manage the date formating as per your requirements
# Mine is in format of YYYYMMdd
def addDate(image):
  img_date = ee.Date(image.date())
  img_date = ee.Number.parse(img_date.format('YYYYMMdd'))
  return image.addBands(ee.Image(img_date).rename('date').toInt())

def getImage(product,start_date,end_date,variable):
  return ee.ImageCollection(product).filter(ee.Filter.date(start_date, end_date)).select(variable)

ndvi=getImage("MODIS/061/MOD13A2",start_date_,end_date_,"NDVI");
sm=getImage("NASA/SMAP/SPL4SMGP/007",start_date_,end_date_,"sm_surface");
preci=getImage("UCSB-CHG/CHIRPS/DAILY",start_date_,end_date_,"precipitation");

# Define a function to calculate the weekly average
def calculatebiWeeklypreci(imageCollection):
  daysInWeek = 16

  # Calculate the number of weeks in the collection
  weeks = ee.List.sequence(0, imageCollection.size().subtract(1).divide(daysInWeek).floor())

  # Function to calculate the weekly average
  def weeklyAverage(week):
    start = ee.Date(imageCollection.first().get('system:time_start')) \
      .advance(ee.Number(week).multiply(daysInWeek), 'day')
    end = start.advance(daysInWeek, 'day')

    weeklyImages = imageCollection \
      .filterDate(start, end);
      #.select('sm_surface'); 

    # Calculate the mean for the week
    #weeklyMean = weeklyImages.reduce(ee.Reducer.mean())
    weeklyMean = weeklyImages.reduce(ee.Reducer.sum())


    # Set the 'system:time_start' property to the start of the week
    startDate = ee.Date(start)
    newImage = weeklyMean.set('system:time_start', startDate.millis())

    return newImage

  return ee.ImageCollection.fromImages(weeks.map(weeklyAverage))


  # Define a function to calculate the weekly average
def calculatebiWeeklysm(imageCollection):
  daysInWeek = 16

  # Calculate the number of weeks in the collection
  weeks = ee.List.sequence(0, imageCollection.size().subtract(1).divide(daysInWeek).floor())

  # Function to calculate the weekly average
  def weeklyAverage(week):
    start = ee.Date(imageCollection.first().get('system:time_start')) \
      .advance(ee.Number(week).multiply(daysInWeek), 'day')
    end = start.advance(daysInWeek, 'day')

    weeklyImages = imageCollection \
      .filterDate(start, end);
      #.select('sm_surface'); 

    # Calculate the mean for the week
    #weeklyMean = weeklyImages.reduce(ee.Reducer.mean())
    weeklyMean = weeklyImages.reduce(ee.Reducer.mean())


    # Set the 'system:time_start' property to the start of the week
    startDate = ee.Date(start)
    newImage = weeklyMean.set('system:time_start', startDate.millis())

    return newImage

  return ee.ImageCollection.fromImages(weeks.map(weeklyAverage))                                       


# Calculate the biweekly average
sm16 = calculatebiWeeklysm(sm)
preci16 = calculatebiWeeklypreci(preci)
#ndvi16 = calculateWeeklyAverage(ndvi)

points = pd.read_excel(grid_points)
columns_to_round = ['X', 'Y']
points[columns_to_round] = points[columns_to_round].round(3)

features=[]
for index, row in points.iterrows():
  poi_geometry = ee.Geometry.Point([row['X'], row['Y']])
  poi_properties = dict(row)
  poi_feature = ee.Feature(poi_geometry, poi_properties)
  features.append(poi_feature)
  ee_fc = ee.FeatureCollection(features)


def rasterExtraction(image):
    feature = image.sampleRegions(
        collection = ee_fc, # feature collection here
        scale = 10000 # Cell size of raster
    )
    return feature

def generateVariables(variable,path):
  result = variable.first().getInfo()
  columns = list(result['properties'].keys())
  nested_list = variable.reduceColumns(ee.Reducer.toList(len(columns)), columns).values().get(0)
  data = nested_list.getInfo()
  df = pd.DataFrame(data, columns=columns)
  df.to_csv(path,mode='w')
  return df;

def mergeDataframes(df1,df2):
  return pd.merge(df1, df2, on=['X', 'Y', 'date'], how='inner')


if ndvi.size().getInfo()>0:
  ndvi1 = ndvi.filterBounds(ee_fc).map(addDate).map(rasterExtraction).flatten()
  ndvi2=generateVariables(ndvi1,ndvi_output);
  if preci16.size().getInfo()>0:
    preci1 = preci16.filterBounds(ee_fc).map(addDate).map(rasterExtraction).flatten()
    preci2=generateVariables(preci1,preci_output);
    if sm16.size().getInfo()>0:
      sm1 = sm16.filterBounds(ee_fc).map(addDate).map(rasterExtraction).flatten()
      sm2=generateVariables(sm1,sm_output);
      merged_df=mergeDataframes(ndvi2,sm2);
      merged_df2=mergeDataframes(merged_df,preci2);
      selected_columns = ['X', 'Y', 'date', 'NDVI', 'sm_surface_mean', 'precipitation_sum']
      result_df = merged_df2[selected_columns].loc[:, ~merged_df2[selected_columns].columns.duplicated()]
      new_column_names = ['lon', 'lat', 'date', 'ndvi', 'sm', 'preci']
      # Rename the columns
      result_df.rename(columns=dict(zip(result_df.columns, new_column_names)), inplace=True)
      result_df.to_csv(combined_output,mode='w')




