import sys,os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from codes.tools import GeoserverClient
from codes.funcs import *
from codes.send_notification import *


print ("")
print ("")
print ("         Importing raster files to the Geoserver ")
print ("")
print ("")
print ("")


# Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')

with open(filepath) as f:
    data = json.load(f)

geo_url = data["geoserve_url"]
geo_user = data["geoserver_user"]
geo_pwd = data["geoserver_pwd"]


workspace_name = "waterpoints_et"
forecast_type = "biomass"
country_iso = workspace_name.split("_")[1]
folder_data = os.path.join(project_root, 'data')
outputs_path = os.path.join(project_root, 'outputs')
folder_layers = os.path.join(folder_data, "layers")
folder_properties = os.path.join(folder_layers, forecast_type+"_properties")
folder_tmp = os.path.join(folder_data, "tmp")
fname = os.path.join(outputs_path, "new_data_list_FINAL.txt")
with open(fname) as f:
    file_content = f.readlines()

rasters_files = [x.strip() for x in file_content]

if len(rasters_files)==0:
    exit_program()

stores_biomass = [forecast_type]

# Connecting
geoclient = GeoserverClient(geo_url, geo_user, geo_pwd)



# uploading biomass
for current_store in stores_biomass:

    current_layer = forecast_type+"_et"
    current_rasters_folder = os.path.join(folder_layers, current_layer)
    #rasters_files = os.listdir(current_rasters_folder)

    store_name = current_store
    print("Importing")
    geoclient.connect()
    geoclient.get_workspace(workspace_name)

    for r in rasters_files:
        store = geoclient.get_store(store_name)
        layer = os.path.join(current_rasters_folder, r)
        if not store:
            print("Creating mosaic")
            geoclient.create_mosaic(
                store_name, layer, folder_properties, folder_tmp)
        else:
            print("Updating mosaic")
            geoclient.update_mosaic(
                store, layer, folder_properties, folder_tmp)


print("Process completed")
subject = "Biomass Updated"
dynamic_header = "Notice: Biomass Data Updated Successfully:"
dynamic_info = "I hope this email finds you well. We are reaching out to inform you Biomass data has been updated successfully. Below, you will find specific details about the raster files uploaded."
dynamic_content = "<p>Raster Files:</p>"
dynamic_content += f"<li><span style='color: green;'>{rasters_files}</span></li>"
send_email_html(email_list, dynamic_header,dynamic_info,dynamic_content,subject) 