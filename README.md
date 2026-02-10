# Biomass ETL

![GitHub release (latest by date)](https://img.shields.io/github/v/release/CIAT-DAPA/lswms_forage_etl) ![](https://img.shields.io/github/v/tag/CIAT-DAPA/lswms_forage_etl)

This ETL is meant to automate the process of Estimating Biomass. The maps can be accessed through [the forage tool ](https://et.waterpointsmonitoring.net/forage).

## Setup and Installation

The installation described here will make use of conda to ensure there are no package conflicts with
existing or future applications on the machine. It is highly recommended using a dedicated environment
for this application to avoid any issues.

### Recommended

Conda (To manage packages within the applications own environment)

### Environment

- Create the env

```commandline
conda env create -f environment.yml
```

Add a file named data.json in the base directory `src/`. This file will hold a json object containing
the secure information needed to run your application. Copy the following object into your file then
edit the values as described in each of the { ... } blocks. This file is in the .gitignore so it will
not be exposed publicly when you push and updates to your repo.

```json
{
  "service_account": "",
  "geoserve_url": "https://domain/geoserver/rest/",
  "geoserver_user": "",
  "geoserver_pwd": "",
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 465,
  "smtp_username": "",
  "smtp_password": "",
  "email_list": [""],
  "trends_url": "https://domain/api/v1/biomass_trend/update",
  "forecasts_url": "https://domain/api/v1/biomass_forecast/update",
  "api_key": "",
  "default_start_date": "2026-01-18",
  "data_latency_days": 2,
  "start_date_override": null,
  "current_date_override": null
}
```

### Google Earth Engine (GEE) authentication (service account)

This project uses a Google Cloud **service account** and a **JSON key** to authenticate to Earth Engine.

1. Create or select a Google Cloud project (note the **Project ID**):  
   https://console.cloud.google.com/projectcreate
2. Enable the **Google Earth Engine API** (ensure the correct project is selected in the top bar):  
   https://console.cloud.google.com/apis/library
3. Register the same Cloud project for Earth Engine:  
   https://code.earthengine.google.com/register
4. Create a service account:  
   https://console.cloud.google.com/iam-admin/serviceaccounts
   - Minimum permission for read-only workflows: **Earth Engine Resource Viewer** (`roles/earthengine.viewer`)
5. Create and download the private key (JSON):
   - Service account → **Keys** → **Add key** → **Create new key** → **JSON**
6. Save the downloaded key as `private_key.json` in the repo base directory `src/`(**do not commit this file** — it is in `.gitignore`), and set `service_account` in `data.json` to the service account email (e.g. `gee-ndvi-sa@<PROJECT_ID>.iam.gserviceaccount.com`).

### Preload existing biomass rasters

Before running the pipeline, make sure **all existing biomass raster files** you want the ETL to use are present in:

- `src/data/layers/biomass_et/` (Windows path: `src\data\layers\biomass_et`)

The order of script execution is:

1. data_extraction_v2.py
2. gwr_model.py
3. rasterize.py
4. import_biomass.py
5. ForecastMain.py

You will need a `private_key.json` file associated with the `service_account` placed in the base directory `src/`. This file is in `.gitignore` so it will not be exposed publicly when you push updates to your repo.

There is master_script.py to run all the other scripts.
