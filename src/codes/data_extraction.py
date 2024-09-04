import sys, os
import json
import ee
from ee.ee_exception import EEException
import numpy as np
import pandas as pd
import datetime
import requests
import xarray as xr
import datetime
from datetime import date, timedelta
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.funcs import *
from codes.send_notification import *

print ("")
print ("")
print ("         Extracting Data from Google Earth Engine ")
print ("")
print ("")
print ("")




# Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')


with open(filepath) as f:
    data = json.load(f)


private_key = os.path.join(project_root, 'private_key.json')
outputs_path = os.path.join(project_root, 'outputs')
inputs_path = os.path.join(project_root, 'inputs')
data_path = os.path.join(project_root, 'data')
layers_path = os.path.join(data_path, "layers")
biomass_path = os.path.join(layers_path, "biomass_et")
grid_points=os.path.join(inputs_path,  "grid_points.xlsx")
sm_output=os.path.join(outputs_path,  "sm.csv")
ndvi_output=os.path.join(outputs_path,  "ndvi.csv")
preci_output=os.path.join(outputs_path,  "preci.csv")
combined_output=os.path.join(outputs_path,  "combined.csv")
tamsat_nc_dir = os.path.join(inputs_path, "tamsat_daily")

service_account = data["service_account"]
email_list= data["email_list"]

def send_error_notification(subject, message):
    truncate_file(combined_output)
    dynamic_header = "Notice: Biomass Data Failed to Update:"
    dynamic_info = "I hope this email finds you well. We are reaching out to inform you about a failure to update the biomass data. Below, you will find specific details about the error."
    dynamic_content = f"<p>Error Message:</p><li><span style='color: red;'>{message}</span></li>"
    send_email_html(email_list, dynamic_header, dynamic_info, dynamic_content, subject)
    exit_program()
    
try:
  credentials = ee.ServiceAccountCredentials(service_account, private_key)
  ee.Initialize(credentials)
except EEException as e:
        send_error_notification("Biomass not updated", "Could not authenticate with the Google Earth Engine")

layers = os.listdir(biomass_path)
if len(layers)>0:
  sorted_layers = sorted(layers, reverse=True)
  latest_image=sorted_layers[0][:-4][8:]
  date_format= '%Y%m%d';
  date_obj = datetime.datetime.strptime(latest_image, date_format)
  start_date_obj =date_obj.date() + timedelta(days=1)
  start_date_ =start_date_obj.strftime('%Y-%m-%d')

else:
  start_date_ = "2024-02-02";
end_date_ = date.today().strftime('%Y-%m-%d')
# manage the date formating as per your requirements
# Mine is in format of YYYYMMdd
def addDate(image):
  try:
    img_date = ee.Date(image.date())
    img_date = ee.Number.parse(img_date.format('YYYYMMdd'))
    return image.addBands(ee.Image(img_date).rename('date').toInt())
  except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))  

def getImage(product,start_date,end_date,variable):
  try:
    return ee.ImageCollection(product).filter(ee.Filter.date(start_date, end_date)).select(variable)
  except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e)) 

ndvi=getImage("MODIS/061/MOD13A2",start_date_,end_date_,"NDVI");
sm=getImage("NASA/SMAP/SPL4SMGP/007",start_date_,end_date_,"sm_surface");
#preci=getImage("UCSB-CHG/CHIRPS/DAILY",start_date_,end_date_,"precipitation");

print("NDVI Collection Size: ", ndvi.size().getInfo())
print("SM Collection Size: ", sm.size().getInfo())
#print("Precipitation Collection Size: ", preci.size().getInfo())

# Define a function to calculate the 16 days GEE precipitation average 
def calculatebiWeeklypreci(imageCollection):
  try:
      daysInWeek = 16
      weeks = ee.List.sequence(0, imageCollection.size().subtract(1).divide(daysInWeek).floor())
      
      def biWeeklySum(week):
          start = ee.Date(imageCollection.first().get('system:time_start')).advance(ee.Number(week).multiply(daysInWeek), 'day')
          end = start.advance(daysInWeek, 'day')
          biWeeklyImages = imageCollection.filterDate(start, end)
          
          biWeeklySum = biWeeklyImages.reduce(ee.Reducer.sum())
          sortedImages = biWeeklyImages.sort('system:time_start', False)
          endDate = ee.Date(sortedImages.first().get('system:time_start'))
          newImage = biWeeklySum.set('system:time_start', endDate.millis())
          return newImage
      
      return ee.ImageCollection.fromImages(weeks.map(biWeeklySum))

  except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))

  # Define a function to calculate the weekly average
def calculatebiWeeklysm(imageCollection):
  try:
      daysInBiWeek = 16
      hoursInDay = 24
      hoursInBiWeek = daysInBiWeek * hoursInDay
      threeHourlyIntervalsInBiWeek = hoursInBiWeek / 3
      numBiWeeks = ee.Number(imageCollection.size()).divide(threeHourlyIntervalsInBiWeek).floor()

      def biWeeklyAverage(week):
          start = ee.Date(imageCollection.first().get('system:time_start')).advance(ee.Number(week).multiply(daysInBiWeek), 'day')
          end = start.advance(daysInBiWeek, 'day')
          biWeeklyImages = imageCollection.filterDate(start, end)
          
          biWeeklyMean = biWeeklyImages.reduce(ee.Reducer.mean())
          sortedImages = biWeeklyImages.sort('system:time_start', False)
          endDate = ee.Date(sortedImages.first().get('system:time_start'))
          newImage = biWeeklyMean.set('system:time_start', endDate.millis())
          return newImage
      
      weeks = ee.List.sequence(0, numBiWeeks.subtract(1))
      return ee.ImageCollection.fromImages(weeks.map(biWeeklyAverage))
  
  except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))                                          




def generate_date_strings(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date.strftime('%Y/%m/rfe%Y_%m_%d')
        current_date += timedelta(days=1)

def download_tamsat_data(date_str, save_path):
    url = f"https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/v3.1/daily/{date_str}.v3.1.nc"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Data for {date_str} downloaded successfully and saved to {save_path}")
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            print(f"Data for {date_str} not found (404). Skipping this date.")
        else:
            print("TAMSAT data download failed", f"HTTP error occurred: {http_err}")
    except Exception as e:
        send_error_notification("Biomass not updated", "TAMSAT data download failed: " + str(e)) 
       

if not os.path.exists(tamsat_nc_dir):
    os.makedirs(tamsat_nc_dir)

start_date_obj = datetime.datetime.strptime(start_date_, '%Y-%m-%d')
end_date_obj = datetime.datetime.strptime(end_date_, '%Y-%m-%d')

for date_str in generate_date_strings(start_date_obj, end_date_obj):
    date_str1=date_str.split('/')[-1]
    save_path = os.path.join(tamsat_nc_dir, f"{date_str1}.nc")
    download_tamsat_data(date_str, save_path)

def read_and_aggregate_tamsat_data(nc_dir, start_date):
    try:
        ds_list = []
        current_date = start_date
        latest_date = None

        while True:
            date_str = current_date.strftime('%Y_%m_%d')
            nc_file = os.path.join(nc_dir, f"rfe{date_str}.nc")
            if os.path.exists(nc_file):
                ds = xr.open_dataset(nc_file)
                ds_list.append(ds)
                latest_date = current_date
                print(f"File {nc_file} found with time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            else:
                break  # Stop if no more files are found
            current_date += timedelta(days=1)

        if not latest_date:
            print("No TAMSAT NetCDF files available.")

        total_days = (latest_date - start_date).days + 1
        num_full_periods = total_days // 16
        adjusted_end_date = start_date + timedelta(days=(num_full_periods * 16 - 1))

        print(f"Total days: {total_days}, Number of full 16-day periods: {num_full_periods}")
        print(f"Adjusted end date for complete periods: {adjusted_end_date}")

        combined_ds = xr.concat(ds_list, dim='time')
        combined_ds = combined_ds.sel(time=slice(start_date, adjusted_end_date))
        print(f"Combined dataset time dimension length: {len(combined_ds.time)}")
        return combined_ds
    except Exception as e:
        send_error_notification("Biomass not updated", "Failed to process TAMSAT data: " + str(e)) 
        

def calculate_biweekly_precipitation(ds):
    try:
        # Determine the number of full 16-day periods
        num_full_periods = len(ds.time) // 16
        print(f"Number of full 16-day periods: {num_full_periods}")

        # Filter the dataset to include only complete periods
        ds = ds.isel(time=slice(0, num_full_periods * 16))
        print(f"Filtered dataset time dimension length: {len(ds.time)}")

        # Resample to biweekly sums, label the right edge, and close on the right
        ds_biweekly = ds.resample(time='16D', label='right', closed='right').sum()
        print(f"Resampled dataset time dimension length: {len(ds_biweekly['time'])}")

        # Calculate last_times and format them to 'YYYYMMDD'
        last_times = [
            np.datetime64(ds.time.values[i * 16 + 15], 'D').astype(str).replace('-', '')
            for i in range(num_full_periods)
        ]
        print(f"Calculated last_times: {last_times}")
        print(f"Length of last_times: {len(last_times)}")

        # Adjust ds_biweekly if extra periods are detected
        if len(ds_biweekly['time']) > len(last_times):
            print(f"Extra period detected. Removing the last period from ds_biweekly.")
            ds_biweekly = ds_biweekly.isel(time=slice(0, len(last_times)))

        if len(last_times) != len(ds_biweekly['time']):
            print(f"Length mismatch: last_times = {len(last_times)}, ds_biweekly['time'] = {len(ds_biweekly['time'])}")
           
        # Assign the formatted time coordinates to ds_biweekly
        ds_biweekly = ds_biweekly.assign_coords(time=('time', last_times))
        print(f"Assigned new time coordinates to ds_biweekly.")

        return ds_biweekly
    except Exception as e:
        send_error_notification("Biomass not updated", "Failed to process TAMSAT data: " + str(e))
        
        

    
def sample_dataset(ds, grid_points_file):
    try:
        points = pd.read_excel(grid_points_file)
        
        samples = []
        for _, row in points.iterrows():
            lon, lat = row['X'], row['Y']
            sample = ds.sel(lat=lat, lon=lon, method='nearest').to_dataframe().reset_index()
            sample['X'] = lon
            sample['Y'] = lat
            sample = sample[['time', 'X', 'Y', 'rfe_filled']]
            sample.rename(columns={'time': 'date', 'rfe_filled': 'precipitation_sum'}, inplace=True)
            
            # Round off the X and Y values
            columns_to_round = ['X', 'Y']
            sample[columns_to_round] = sample[columns_to_round].round(3)
            
            samples.append(sample)
        return pd.concat(samples)
    except Exception as e:
        send_error_notification("Biomass not updated", "Failed to process TAMSAT data: " + str(e))


# Calculate the soil moisture 16 days  average
sm16 = calculatebiWeeklysm(sm)

try:
    tamsat_ds = read_and_aggregate_tamsat_data(tamsat_nc_dir, start_date_obj)
    biweekly_precipitation = calculate_biweekly_precipitation(tamsat_ds)

  
    preci_sampled = sample_dataset(biweekly_precipitation, grid_points)
    preci_sampled.to_csv(preci_output, index=False)

except Exception as e:
    send_error_notification("Biomass not updated", "Failed to process TAMSAT data: " + str(e))



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
  
try:
  if ndvi.size().getInfo()>0:
    ndvi1 = ndvi.filterBounds(ee_fc).map(addDate).map(rasterExtraction).flatten()
    ndvi2=generateVariables(ndvi1,ndvi_output);
  else:
    send_error_notification("Biomass not updated", "Latest Terra Vegetation Indices 16-Day Global 1km products cannot be found in  Google Earth Engine") 
  
except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e)) 
          

try:
  if sm16.size().getInfo()>0:
    sm1 = sm16.filterBounds(ee_fc).map(addDate).map(rasterExtraction).flatten()
    sm2=generateVariables(sm1,sm_output);
    merged_df=mergeDataframes(ndvi2,sm2);
    merged_df['date'] = merged_df['date'].astype(str)
    preci_sampled['date'] = preci_sampled['date'].astype(str)
    merged_df2=mergeDataframes(merged_df,preci_sampled);
    selected_columns = ['X', 'Y', 'date', 'NDVI', 'sm_surface_mean', 'precipitation_sum']
    result_df = merged_df2[selected_columns].loc[:, ~merged_df2[selected_columns].columns.duplicated()]
    new_column_names = ['lon', 'lat', 'date', 'ndvi', 'sm', 'preci']
    # Rename the columns
    result_df.rename(columns=dict(zip(result_df.columns, new_column_names)), inplace=True)
    result_df.to_csv(combined_output,mode='w')
  else:
    send_error_notification("Biomass not updated", "Latest Soil Moisture products cannot be found in  Google Earth Engine") 
  
except EEException as e:
       send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))    




