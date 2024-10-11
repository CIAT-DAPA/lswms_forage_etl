import sys, os
import json
import ee
from ee.ee_exception import EEException
import numpy as np
import pandas as pd
import datetime
import requests
import xarray as xr
from datetime import datetime, timedelta, date

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.funcs import *
from codes.send_notification import *

print ("")
print ("")
print ("         Extracting Data from Google Earth Engine ")
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
grid_points=os.path.join(inputs_path, "grid_points.xlsx")
aoi_path=os.path.join(inputs_path, "aoi.geojson")
sm_output=os.path.join(outputs_path, "sm.csv")
ndvi_output=os.path.join(outputs_path, "ndvi.csv")
preci_output=os.path.join(outputs_path, "preci.csv")
combined_output=os.path.join(outputs_path, "combined.csv")
tamsat_nc_dir = os.path.join(inputs_path, "tamsat_daily")

service_account = data["service_account"]
email_list= data["email_list"]

with open(aoi_path) as f:
    aoi_data = json.load(f)

  

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



coordinates = aoi_data['geometry']['coordinates']
aoi_ee = ee.Geometry.Polygon(coordinates)


from datetime import timedelta


# Function to get 16-day composite periods
# Function to get 16-day composite periods
def get_composite_periods(start_date_str, current_date_str):
    # Convert date strings to datetime.date objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()
    
    periods = []
    
    # Initialize the starting point as the provided start_date
    current_period_start = start_date
    
    # Continue generating 16-day periods until current_date is reached or exceeded
    while current_period_start <= current_date:
        # Define the end of the current period as 16 days after the start
        current_period_end = current_period_start + timedelta(days=15)
        
        # Check if the current period is a full 16-day period
        if current_period_end <= current_date:
            # Append the current 16-day period since it's a full period
            periods.append({
                'start': current_period_start,
                'end': current_period_end
            })
        else:
            # Break the loop if the period end exceeds the current date (not a full period)
            break
        
        # Move to the next period start (the day after the current period's end)
        current_period_start = current_period_end + timedelta(days=1)
    
    return periods









def addDate(image):
    try:
        img_date = ee.Date(image.date())
        img_date = ee.Number.parse(img_date.format('YYYYMMdd'))
        return image.addBands(ee.Image(img_date).rename('date').toInt())
    except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))


def getImage(product,start_date,end_date,variable):
  try:
    return ee.ImageCollection(product).filter(ee.Filter.date(start_date, end_date)).filterBounds(aoi_ee).select(variable)
  except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e)) 

# Function to calculate NDVI from Sentinel-2 and composite over 10-day periods
def compositeVIIRSNDVI(start_date, end_date):
    try:
        # Convert datetime.date objects to strings in the format 'YYYY-MM-DD'
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Define the VIIRS collection with filters
        collection = ee.ImageCollection("NASA/VIIRS/002/VNP09GA") \
            .filter(ee.Filter.date(start_date_str, end_date_str)) \
            .filterBounds(aoi_ee)  # Adjust aoi_ee as per your area of interest

        # Function to calculate NDVI using VIIRS bands (M7 = NIR, M5 = Red)
        def calc_ndvi(image):
            ndvi = image.normalizedDifference(['M7', 'M5']).rename('NDVI')
            return ndvi.set('system:time_start', image.get('system:time_start'))

        # Function to mask clouds using the QF1 band bitmask
        def mask_clouds(image):
            qf1 = image.select('QF1')

            # Extract bits 0-1 for cloud mask quality (Medium = 2, High = 3)
            cloud_quality = qf1.bitwiseAnd(3)  # Extract bits 0-1
            cloud_quality_mask = cloud_quality.gte(2)  # Allow Medium or High quality

            # Extract bits 2-3 for cloud detection confidence (Confident clear = 0, Probably clear = 1)
            cloud_confidence = qf1.rightShift(2).bitwiseAnd(3)  # Extract bits 2-3
            cloud_confidence_mask = cloud_confidence.lte(1)  # Allow Confident clear or Probably clear

            # Combine both masks: cloud_quality_mask AND cloud_confidence_mask
            cloud_mask = cloud_quality_mask.And(cloud_confidence_mask)

            # Return the masked image
            return image.updateMask(cloud_mask)

        # Apply cloud masking and NDVI calculation to each image in the collection
        ndvi_collection = collection.map(mask_clouds).map(calc_ndvi)

        print("NDVI Collection Size: ", ndvi_collection.size().getInfo())

        # Ensure that the period end does not exceed the actual end date provided
        period_end = ee.Date(end_date_str)

        # Filter NDVI collection for the given period and calculate the mean
        period_collection = ndvi_collection.filterDate(start_date_str, end_date_str)
        composite_image = period_collection.reduce(ee.Reducer.mean()).set('system:time_start', period_end.millis())

        return composite_image

    except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))







# TAMSAT Data Download and Processing Functions
def generate_date_strings(start_date, end_date):
    # Ensure that both start_date and end_date are datetime.date objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Initialize current_date to start_date
    current_date = start_date

    # Generate date strings until current_date exceeds end_date
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



# Function to read and aggregate TAMSAT data for 10-day periods

def read_and_aggregate_tamsat_data(nc_dir, start_date, end_date):
    try:
        # Ensure start_date and end_date are datetime objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        ds_list = []
        current_date = start_date
        latest_date = None

        while current_date <= end_date:
            # Convert current_date to the required string format
            date_str = current_date.strftime('%Y_%m_%d')
            nc_file = os.path.join(nc_dir, f"rfe{date_str}.nc")
            
            if os.path.exists(nc_file):
                ds = xr.open_dataset(nc_file)
                ds_list.append(ds)
                latest_date = current_date
                print(f"File {nc_file} found with time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            
            # Move to the next day
            current_date += timedelta(days=1)

        if not ds_list:
            print("No TAMSAT NetCDF files available for the given period.")
            return None

        # Concatenate the dataset list along the time dimension
        combined_ds = xr.concat(ds_list, dim='time')
        combined_ds = combined_ds.sel(time=slice(start_date, end_date))
        print(f"Combined dataset time dimension length: {len(combined_ds.time)}")
        
        return combined_ds

    except Exception as e:
        send_error_notification("Biomass not updated", f"Failed to process TAMSAT data: {str(e)}")



def calculatebiWeeklysm(imageCollection, start_date, end_date):
    try:
        # Convert start_date and end_date to strings for EE
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Convert to Earth Engine date objects
        start_date_ee = ee.Date(start_date_str)
        end_date_ee = ee.Date(end_date_str)
        
        # Filter soil moisture collection for the entire period
        period_collection = imageCollection.filterDate(start_date_ee, end_date_ee)
        
        # Calculate the mean image for the period
        biWeeklyMean = period_collection.reduce(ee.Reducer.mean())

        # Set the system time of the mean image to the end of the period
        biWeeklyMean = biWeeklyMean.set('system:time_start', end_date_ee.millis())

        return biWeeklyMean
    
    except EEException as e:
        send_error_notification("Biomass not updated", "Google Earth Engine Error: " + str(e))



# Function to calculate 10-day precipitation sums from TAMSAT data

def calculate_10day_precipitation(ds, end_date):
    try:
        # Check if the dataset has any time entries
        if 'time' not in ds or len(ds.time) == 0:
            print("No data available for the specified period.")
            return None

        # Sum precipitation over the entire time dimension (since it's already filtered)
        ds_10day = ds.sum(dim='time', skipna=True)
        print("Dataset summed over time.")

        # Ensure that end_date is properly formatted as a string
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Assign the last date as the time coordinate (convert to 'YYYYMMDD' format)
        last_time = np.datetime64(end_date, 'D').astype(str).replace('-', '')
        print(f"Assigned new time coordinate: {last_time}")

        # Reassign the 'time' coordinate with the last_time value
        ds_10day = ds_10day.expand_dims('time').assign_coords(time=[last_time])
        print("Time dimension added back to dataset.")

        return ds_10day

    except Exception as e:
        send_error_notification("Biomass not updated", f"Failed to process TAMSAT data: {str(e)}")



# Function to sample TAMSAT data
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




# Sample grid points
points = pd.read_excel(grid_points)
columns_to_round = ['X', 'Y']
points[columns_to_round] = points[columns_to_round].round(3)

features = []
for index, row in points.iterrows():
    poi_geometry = ee.Geometry.Point([row['X'], row['Y']])
    poi_properties = dict(row)
    poi_feature = ee.Feature(poi_geometry, poi_properties)
    features.append(poi_feature)
    ee_fc = ee.FeatureCollection(features)

def rasterExtraction(image):
    feature = image.sampleRegions(
        collection = ee_fc,  # Feature collection here
        scale = 10000  # Cell size of raster
    )
    return feature

def generateVariables(variable):
    result = variable.first().getInfo()
    columns = list(result['properties'].keys())
    nested_list = variable.reduceColumns(ee.Reducer.toList(len(columns)), columns).values().get(0)
    data = nested_list.getInfo()
    df = pd.DataFrame(data, columns=columns)
    return df

def mergeDataframes(df1, df2):
    return pd.merge(df1, df2, on=['X', 'Y', 'date'], how='inner')


layers = os.listdir(biomass_path)
if len(layers) > 0:
    sorted_layers = sorted(layers, reverse=True)
    latest_image = sorted_layers[0][:-4][8:]
    date_format = '%Y%m%d'
    date_obj = datetime.strptime(latest_image, date_format)
    start_date_obj = date_obj.date() + timedelta(days=1)
    start_date_ = start_date_obj.strftime('%Y-%m-%d')
else:
    start_date_ = "2024-02-02"


current_date = date.today().strftime('%Y-%m-%d')
#current_date = "2024-10-01"  # Current date (e.g., 2024-10-05)



# Convert start_date_ to a datetime.date object
start_date_obj = datetime.strptime(start_date_, '%Y-%m-%d').date()

# Use current_date directly (no need to convert it)
end_date_obj = current_date
#end_date_obj = datetime.strptime(current_date, '%Y-%m-%d').date()


for date_str in generate_date_strings(start_date_obj, end_date_obj):
    date_str1 = date_str.split('/')[-1]
    save_path = os.path.join(tamsat_nc_dir, f"{date_str1}.nc")
    download_tamsat_data(date_str, save_path)

# Get the composite periods based on current date
composite_periods = get_composite_periods(start_date_, current_date)

ndvi_combined = pd.DataFrame()
sm_combined = pd.DataFrame()
preci_combined = pd.DataFrame()

# Example of how to loop through these periods and run your processing functions
for period in composite_periods:
    start_date = period['start']
    end_date = period['end']
    print(f"Processing data from {start_date} to {end_date}")
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # NDVI processing
    ndvi_10day =  compositeVIIRSNDVI(start_date, end_date)

    # Check if ndvi_10day is not empty
    if ndvi_10day:
        # Apply addDate and rasterExtraction directly to the single image
        ndvi1 = addDate(ndvi_10day)
        ndvi1_extracted = rasterExtraction(ndvi1)
        
        # Generate variables and append the results to the combined dataframe
        ndvi_sampled = generateVariables(ndvi1_extracted)
        ndvi_combined = pd.concat([ndvi_combined, ndvi_sampled], ignore_index=True)
    else:
        send_error_notification("Biomass not updated", "Latest Sentinel-2 NDVI products cannot be found in Google Earth Engine")

    sm = getImage("NASA/SMAP/SPL4SMGP/007", start_date_str, end_date_str, "sm_surface")
    print("SM Collection Size: ", sm.size().getInfo())
    sm_10day = calculatebiWeeklysm(sm,start_date, end_date)
    # Since sm_10day is now an ee.Image, you can apply rasterExtraction and addDate directly to the image
    if sm_10day:
        # Apply addDate and rasterExtraction directly to the sm_10day image
        sm1 = addDate(sm_10day)
        sm1_extracted = rasterExtraction(sm1)
        sm_sampled = generateVariables(sm1_extracted)
        sm_combined = pd.concat([sm_combined, sm_sampled], ignore_index=True)
    else:
        send_error_notification("Biomass not updated", "Soil moisture data is missing for the period.")
    try:
        tamsat_ds = read_and_aggregate_tamsat_data(tamsat_nc_dir, start_date_str,end_date_str)
        biweekly_precipitation = calculate_10day_precipitation(tamsat_ds,end_date_str)
        preci_sampled = sample_dataset(biweekly_precipitation, grid_points)
        preci_combined = pd.concat([preci_combined, preci_sampled], ignore_index=True)
    except Exception as e:
        send_error_notification("Biomass not updated", "Failed to process TAMSAT data: " + str(e))


# Save combined results
ndvi_combined.to_csv(ndvi_output, index=False)
sm_combined.to_csv(sm_output, index=False)
preci_combined.to_csv(preci_output, index=False)

# Merge and save final combined CSV
try:
    merged_df = mergeDataframes(ndvi_combined, sm_combined)
    merged_df['date'] = merged_df['date'].astype(str)
    preci_combined['date'] = preci_combined['date'].astype(str)
    merged_df2 = mergeDataframes(merged_df, preci_combined)
    result_df = merged_df2[['X', 'Y', 'date', 'NDVI_mean', 'sm_surface_mean', 'precipitation_sum']]
    result_df.columns = ['lon', 'lat', 'date', 'ndvi', 'sm', 'preci']
    result_df.to_csv(combined_output, index=False)
    print("Combined CSV saved successfully.")
except Exception as e:
    send_error_notification("Biomass not updated", f"Failed to merge and save final output: {str(e)}")